#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of metamodel as GPE, using the GPyTorch library
"""

import os
import warnings
import copy
import functools
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, RQKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- SAFE TORCH LOAD PATCH ---
# This ensures that torch objects are safely loaded to the CPU if no GPU is available,
# which is critical when using joblib.load on surrogate engines.
if not hasattr(torch, '_original_load'):
    torch._original_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            # Default to CPU if no GPU is available to prevent loading errors
            kwargs['map_location'] = torch.device('cpu') if not torch.cuda.is_available() else None
        return torch._original_load(*args, **kwargs)
    torch.load = _safe_torch_load
# -----------------------------


from .meta_model import (
    MetaModel,
    _preprocessing_fit,
    _bootstrap_fit,
    _preprocessing_eval,
    _bootstrap_eval,
)

warnings.filterwarnings("ignore")
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


class MyExactGPyModel(gpytorch.models.ExactGP):
    """
    Instance of GPyTorch's "ExactGP" library, with custom likelihood, kernel, training points.
    """
    def __init__(self, train_x, train_y, kernel, likelihood):
        super(MyExactGPyModel, self).__init__(train_inputs=train_x, train_targets=train_y, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPEGPy(MetaModel):
    """
    GP MetaModel using the GPyTorch library.
    """

    def __init__(
        self,
        input_obj,
        meta_model_type="GPE",
        gpe_reg_method="adam",
        loss_function="exact",
        training_iter=100,
        n_restarts=10,
        auto_select=False,
        kernel_type="RBF",
        isotropy=True,
        noisy=False,
        nugget=1e-5, # Increased default nugget for better matrix conditioning
        lr=0.05,     # Slightly higher LR since we are using multiple restarts
        scheduler_data=None,
        n_bootstrap_itrs=1,
        normalize_x_method="norm",
        norm_y=True,
        dim_red_method="no",
        verbose=False,
        input_transform="user"
    ):

        is_gaussian = self.check_is_gaussian()

        super().__init__(
            input_obj=input_obj,
            meta_model_type=meta_model_type,
            n_bootstrap_itrs=n_bootstrap_itrs,
            dim_red_method=dim_red_method,
            is_gaussian=is_gaussian,
            verbose=verbose,
            input_transform=input_transform,
        )

        self.meta_model_type = meta_model_type
        self._gpe_reg_method = gpe_reg_method
        self.regression_dict = {}
        self._gpe_reg_options = {}
        self.loss = loss_function
        self.training_iter = training_iter

        self._auto_select = auto_select
        self._kernel_isotropy = isotropy
        self._kernel_noise = noisy
        self._kernel_type = kernel_type
        self._nugget = nugget
        self.n_restarts = n_restarts
        self.lr = lr
        self.normalize_x_method = normalize_x_method
        self.norm_y = norm_y
        self.scheduler_data = scheduler_data

        self._gp_poly = self.AutoVivification()
        self._x_scaler = self.AutoVivification()
        self._bme_score = self.AutoVivification()
        self._kernel_name_dict = self.AutoVivification()
        self._train_loss = self.AutoVivification()
        self._y_scaler = self.AutoVivification()

    def check_is_gaussian(self) -> bool:
        return True

    def build_metamodel(self) -> None:
        pass # Initialization is handled safely in __init__ now via AutoVivification

    @staticmethod
    def convert_to_tensor(array):
        return torch.tensor(array, dtype=torch.float64)
    
    def build_kernels_old(self):
        """
        Initializes kernels WITH STRICT PRIORS to prevent flat-lining.
        """
        _ndim = self.input_space.ndim
        ard_num_dims = _ndim if not self._kernel_isotropy else None

        # # STRICT PRIORS: These force the optimizer to look for realistic physical shapes
        # # GammaPrior(3.0, 6.0) creates a strong preference for lengthscales around 0.5 
        # # (assuming normalized X) and severely punishes values approaching infinity.
        # ls_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        # os_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
        bounds = gpytorch.constraints.Interval(1e-3, 1e3)

        rbf_kernel = ScaleKernel(
            RBFKernel(ard_num_dims=ard_num_dims, lengthscale_constraint=bounds)
        )
        
        matern_kernel = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, lengthscale_constraint=bounds)
        )
        
        rq_kernel = ScaleKernel(
            RQKernel(alpha=1.0, ard_num_dims=ard_num_dims, lengthscale_constraint=bounds)
        )

        kernel_dict = {"RBF": rbf_kernel, "Matern": matern_kernel, "RQ": rq_kernel}

        if self._auto_select:
            kernel_list = list(kernel_dict.values())
            kernel_names = list(kernel_dict.keys())
        else:
            if self._kernel_type not in kernel_dict:
                raise AttributeError(f"The kernel option {self._kernel_type} is not available.")
            kernel_list = [kernel_dict[self._kernel_type]]
            kernel_names = [self._kernel_type]

        return kernel_list, kernel_names

    def build_kernels(self, batch_shape=torch.Size([])):
        """
        Initializes kernels with strict priors and batch shape support for 
        simultaneous multi-output training.
        """
        _ndim = self.input_space.ndim
        ard_num_dims = _ndim if not self._kernel_isotropy else None

        # Soft priors prevent parameters from flat-lining at extreme bounds
        ls_prior = gpytorch.priors.GammaPrior(3.0, 6.0) 
        os_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
        bounds = gpytorch.constraints.Interval(1e-4, 1e3)

        rbf_kernel = ScaleKernel(
            RBFKernel(
                ard_num_dims=ard_num_dims, 
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=bounds
            ),
            batch_shape=batch_shape,
            outputscale_prior=os_prior
        )
        
        matern_kernel = ScaleKernel(
            MaternKernel(
                nu=2.5, 
                ard_num_dims=ard_num_dims, 
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=bounds
            ),
            batch_shape=batch_shape,
            outputscale_prior=os_prior
        )

        kernel_dict = {"RBF": rbf_kernel, "Matern": matern_kernel}

        if self._auto_select:
            return list(kernel_dict.values()), list(kernel_dict.keys())
        else:
            return [kernel_dict[self._kernel_type]], [self._kernel_type]
        
    def transform_x(self, X: np.array, transform_type=None):
        if transform_type is None:
            transform_type = self.normalize_x_method

        if transform_type is None or transform_type.lower() == "none":
            return X, None
        
        if transform_type.lower() == "norm":
            scaler = MinMaxScaler()
        elif transform_type.lower() == "standard":
            scaler = StandardScaler()
        else:
            raise AttributeError(f"No scaler {transform_type} found.")
            
        return scaler.fit_transform(X), scaler

    @_preprocessing_fit
    @_bootstrap_fit
    def fit(self, X: np.array, y: dict, parallel=False, verbose=False, b_i=0):
        
        # # HPC Optimization: Read Slurm allocation safely
        # slurm_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        
        # # HPC Memory Safety Net: High dimensions need more RAM per core
        # if X.shape[1] >= 40 and slurm_cores > 2:
        #     n_workers = max(1, slurm_cores // 2)
        # else:
        #     n_workers = slurm_cores

        # if self.verbose and self.n_bootstrap_itrs == 1:
        #     items = tqdm(y.items(), desc="Fitting regression")
        # else:
        #     items = y.items()
        
        x_scaled, scaler = self.transform_x(X=X)
        self._x_scaler[f"b_{b_i + 1}"] = scaler

        items = tqdm(y.items(), desc="Fitting Batch GPs") if self.verbose else y.items()

        for key, output_matrix in items:
            # output_matrix shape is (n_samples, m_outputs)
            # We train ALL m_outputs simultaneously in one batch!
            result = self.batch_adaptive_regression(x_scaled, output_matrix, verbose=self.verbose)

            # Store the single Batch GP model for this key
            self._gp_poly[f"b_{b_i + 1}"][key] = result["gp"]
            self._bme_score[f"b_{b_i + 1}"][key] = result["bme"]
            self._kernel_name_dict[f"b_{b_i + 1}"][key] = result["kernel_name"]
            self._y_scaler[f"b_{b_i + 1}"][key] = result["y_scaler"]

            try: 
                self._train_loss[f"b_{b_i + 1}"][key] = result["training_loss"]
            except KeyError:    
                pass
    

        # for key, output in items:
        #     out = None
        #     if parallel:  
        #         # Optimized Parallel execution using loky
        #         out = Parallel(n_jobs=n_workers, backend="loky", batch_size="auto")(
        #             delayed(self.adaptive_regression)(
        #                 x_scaled, output[:, idx], idx, self.verbose
        #             )
        #             for idx in range(output.shape[1])
        #         )
        #     else:  
        #         results = map(
        #             functools.partial(self.adaptive_regression, x_scaled, verbose=self.verbose),
        #             [output[:, idx] for idx in range(output.shape[1])],
        #             range(output.shape[1]),
        #         )
        #         out = list(results)

        #     # Store the trained attributes
        #     for i in range(output.shape[1]):
        #         self._gp_poly[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["gp"]
        #         self._bme_score[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["bme"]
        #         self._kernel_name_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["kernel_name"]
        #         self._train_loss[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["training_loss"]
        #         self._y_scaler[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["y_scaler"]

    # ----------------------------------------------------------------------------
    def adaptive_regression(self, X, y, var_idx, verbose=False):
        """
        Trains a single Gaussian Process Emulator (GPE) using PyTorch and GPyTorch.
        This method allows for kernel optimization and can evaluate multiple kernel architectures (e.g., RBF, Matern) and selects 
        the architecture that maximizes the Bayesian Marginal Likelihood (BME) for the given output variable.
        It also allows for different optimizers (Adam or L-BFGS) and includes early stopping for Adam to prevent overfitting.
        """

        # CRITICAL: Prevent PyTorch from hanging in Loky workers
        torch.set_num_threads(1)
        
        # Auto-detect CUDA for massive speedup on GPU nodes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gp_list = {}
        bme = []
        training_loss = []

        if self.norm_y:
            y_scaler = StandardScaler()
            train_y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten() 
        else:
            y_scaler = None
            train_y = y

        kernel_list, kernel_names = self.build_kernels()

        X_tensor = self.convert_to_tensor(X).to(device)
        y_tensor = self.convert_to_tensor(train_y).to(device)

        for i, kernel in enumerate(kernel_list):
            
            kernel = kernel.to(device)

            if self._kernel_noise: 
                # Prior on noise prevents it from explaining 100% of the variance
                # noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
                # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior).to(device)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0)
                ).to(device).double()
            else:
                ns = torch.ones(1, device=device) * self._nugget
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=ns, noise_constraint=gpytorch.constraints.GreaterThan(1e-10)
                ).to(device).double()

            gp = MyExactGPyModel(X_tensor, y_tensor, kernel, likelihood).to(device).double()

            if 'exact' in self.loss.lower():
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
            elif 'loo' in self.loss.lower():
                mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, gp)

            # --- MULTIPLE RESTARTS LOOP ---
            best_loss = float('inf')
            best_model_state = None
            best_loss_list = []

            if self._gpe_reg_method.lower() == "lbfgs":
                
                # 1. L-BFGS OPTIMIZER (Mimics scikit-learn)
                optimizer = torch.optim.LBFGS(
                    gp.parameters(), 
                    lr=self.lr, 
                    max_iter=self.training_iter, # L-BFGS handles its own internal iterations
                    line_search_fn='strong_wolfe' 
                )

                # PyTorch L-BFGS requires a closure function to re-evaluate the loss
                def closure():
                    optimizer.zero_grad()
                    output = gp(X_tensor)
                    loss = -mll(output, y_tensor)
                    loss.backward()
                    return loss

                for restart in range(self.n_restarts):
                    # Randomly perturb hyperparameters for new start locations
                    if restart > 0:
                        with torch.no_grad():
                            gp.covar_module.base_kernel.lengthscale = torch.rand_like(gp.covar_module.base_kernel.lengthscale) * 2.0 + 0.1
                            gp.covar_module.outputscale = torch.rand_like(gp.covar_module.outputscale) * 2.0 + 0.1
                            if self._kernel_noise:
                                gp.likelihood.noise = torch.rand_like(gp.likelihood.noise) * 0.1 + 1e-4
                    
                    # One optimizer step handles up to `max_iter` internal steps
                    optimizer.step(closure)
                    
                    # Evaluate the final loss of this restart
                    with torch.no_grad():
                        current_loss = -mll(gp(X_tensor), y_tensor).item()
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_state = copy.deepcopy(gp.state_dict())
                        best_loss_list = [current_loss] # L-BFGS doesn't easily expose step-by-step loss history

            else:
                # 2. ADAM OPTIMIZER (Best for large datasets/GPU)
                for restart in range(self.n_restarts):
                    
                    if restart > 0:
                        with torch.no_grad():
                            gp.covar_module.base_kernel.lengthscale = torch.rand_like(gp.covar_module.base_kernel.lengthscale) * 2.0 + 0.1
                            gp.covar_module.outputscale = torch.rand_like(gp.covar_module.outputscale) * 2.0 + 0.1
                            if self._kernel_noise:
                                gp.likelihood.noise = torch.rand_like(gp.likelihood.noise) * 0.1 + 1e-4

                    optimizer = torch.optim.Adam(gp.parameters(), lr=self.lr)
                    
                    if self.scheduler_data is not None:
                        step_size = self.scheduler_data.get('step_size', 50)
                        gamma = self.scheduler_data.get('gamma', 0.5)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

                    # Early Stopping Parameters
                    patience = 30
                    min_delta = 1e-3
                    best_iter_loss = float('inf')
                    patience_counter = 0
                    loss_list = []

                    gp.train()
                    likelihood.train()

                    for j in range(self.training_iter):
                        optimizer.zero_grad()
                        output = gp(X_tensor)
                        loss = -mll(output, y_tensor)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(gp.parameters(), max_norm=1.0, error_if_nonfinite=True)
                        optimizer.step()
                        
                        if self.scheduler_data is not None:
                            scheduler.step()
                            
                        current_loss = loss.item()
                        loss_list.append(current_loss)
                        
                        if current_loss < best_iter_loss - min_delta:
                            best_iter_loss = current_loss
                            best_iter_state = copy.deepcopy(gp.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= patience:
                            break
                    
                    # Keep the best local optimum across restarts
                    if best_iter_loss < best_loss:
                        best_loss = best_iter_loss
                        best_model_state = best_iter_state
                        best_loss_list = copy.deepcopy(loss_list)

            # --- LOAD BEST WEIGHTS ---
            gp.load_state_dict(best_model_state)

            # Evaluate BME with best weights
            gp.eval()
            likelihood.eval()
            with torch.no_grad():
                output = gp(X_tensor)
                bme.append(mll(output, y_tensor).item())

            gp_list[i] = gp.cpu() # Move back to CPU for serialization
            training_loss.append(best_loss_list)

        idx_max = np.argmax(bme)
        gp = gp_list[idx_max]

        if var_idx is not None and verbose:
            gp_score = bme[idx_max]
            print("=" * 50)
            print(f"Output variable {var_idx}: converged using {kernel_names[idx_max]} Kernel")
            print(f"BME Score: {gp_score:.3f}")
            print("=" * 50)

        return {
            "gp": gp,
            "bme": bme[idx_max],
            "kernel_name": kernel_names[idx_max],
            "training_loss": training_loss[idx_max],
            "y_scaler": y_scaler
        }

    def batch_adaptive_regression(self, X, y_matrix, verbose=False):
        """
        Adaptively trains a batch of Gaussian Process Emulators (GPEs) simultaneously 
        using PyTorch hardware acceleration.

        This method leverages GPyTorch's Batch Mode to train `M` independent Gaussian Processes 
        (one for each output dimension) at the exact same time. It evaluates multiple kernel 
        architectures (e.g., RBF, Matern) and selects the architecture that maximizes the 
        total Bayesian Marginal Likelihood (BME) across all `M` tasks.

        Note on Batching Mechanics:
        ---------------------------
        While the selected Kernel Architecture (e.g., Matern) is applied globally to all `M` 
        outputs, the underlying hyperparameters (lengthscales, outputscales, and noise) are 
        learned completely independently for each output task.

        Parameters
        ----------
        X : np.ndarray of shape (N_samples, D_dims)
            The input feature matrix. These values should already be scaled/normalized 
            by the class's `transform_x` method prior to entering this function.
        y_matrix : np.ndarray of shape (N_samples, M_tasks)
            The target values for the experimental design. Represents `M` different 
            simulation outputs/components that need to be emulated.
        verbose : bool, optional
            If True, prints a summary of the winning kernel and its BME score upon 
            convergence. The default is False.

        Returns
        -------
        dict
            A dictionary containing the optimized batch model and its metadata:
            - "gp" (gpytorch.models.ExactGP): The trained GPyTorch batch model, 
              moved to the CPU for safe serialization.
            - "bme" (float): The summed Log Marginal Likelihood of the winning 
              kernel across all M tasks.
            - "kernel_name" (str): The name of the winning kernel architecture.
            - "y_scaler" (Scikit-Learn Scaler or None): The scaler object used to 
              standardize the `y_matrix`. Required for inverse-transforming predictions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        N_samples = y_matrix.shape[0]
        M_tasks = y_matrix.shape[1] # Number of simultaneous GPs
        D_dims = X.shape[1]
        
        batch_shape = torch.Size([M_tasks])

        # 1. Scale Y independently per task
        if self.norm_y:
            y_scaler = StandardScaler()
            train_y = y_scaler.fit_transform(y_matrix) 
        else:
            y_scaler = None
            train_y = y_matrix

        # 2. Shape Magic for Batch Mode:
        # X goes from (N, D) -> (M, N, D)
        X_tensor = self.convert_to_tensor(X).to(device)
        X_batch = X_tensor.unsqueeze(0).expand(M_tasks, N_samples, D_dims)
        
        # Y goes from (N, M) -> (M, N)
        Y_batch = self.convert_to_tensor(train_y).t().contiguous().to(device)

        kernel_list, kernel_names = self.build_kernels(batch_shape=batch_shape)
        
        best_overall_bme = -float('inf')
        best_overall_gp = None
        best_overall_kernel_name = None

        for i, kernel in enumerate(kernel_list):
            kernel = kernel.to(device)

            if self._kernel_noise: 
                noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    batch_shape=batch_shape,
                    noise_prior=noise_prior,
                    noise_constraint=gpytorch.constraints.Interval(1e-5, 1.0)
                ).to(device).double()
            else:
                ns = torch.ones(M_tasks, 1, device=device) * self._nugget
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=ns, 
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-10)
                ).to(device).double()

            # Initialize Model
            gp = MyExactGPyModel(X_batch, Y_batch, kernel, likelihood).to(device).double()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

            # Restarts & Adam Setup
            best_loss = float('inf')
            best_model_state = None

            for restart in range(self.n_restarts):
                if restart > 0:
                    # Randomize parameters across the batch for new start
                    with torch.no_grad():
                        gp.covar_module.base_kernel.lengthscale = torch.rand_like(gp.covar_module.base_kernel.lengthscale) * 2.0 + 0.1
                        gp.covar_module.outputscale = torch.rand_like(gp.covar_module.outputscale) * 2.0 + 0.1

                optimizer = torch.optim.Adam(gp.parameters(), lr=self.lr)
                
                gp.train()
                likelihood.train()

                for j in range(self.training_iter):
                    optimizer.zero_grad()
                    output = gp(X_batch)
                    
                    # loss is shape (M,). Summing it optimizes all M tasks simultaneously!
                    loss = -mll(output, Y_batch).sum() 
                    loss.backward()
                    
                    optimizer.step()
                    
                    current_loss = loss.item()
                    
                    # Store best state
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_model_state = copy.deepcopy(gp.state_dict())

            # Load the best weights from restarts
            gp.load_state_dict(best_model_state)
            
            # Evaluate BME
            gp.eval()
            likelihood.eval()
            with torch.no_grad():
                # BME sum across all tasks for this kernel architecture
                bme_score = mll(gp(X_batch), Y_batch).sum().item()

            if bme_score > best_overall_bme:
                best_overall_bme = bme_score
                best_overall_kernel_name = kernel_names[i]
                best_overall_gp = gp.cpu() # Only move the WINNER to CPU!

        if verbose:
            print(f"Batch trained {M_tasks} GPs simultaneously using {best_overall_kernel_name} Kernel. BME: {best_overall_bme:.3f}")

        return {
            "gp": best_overall_gp,
            "bme": best_overall_bme,
            "kernel_name": best_overall_kernel_name,
            "y_scaler": y_scaler
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def scale_x(X: np.array, transform_obj: object):
        if transform_obj is None:
            return X
        return transform_obj.transform(X)

    # Helper method for parallel evaluation
    def _evaluate_single_gp(self, gp, x_tensor, y_scaler=None):
        torch.set_num_threads(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        gp = gp.to(device)
        x_tensor = x_tensor.to(device)
        likelihood_ = gp.likelihood.to(device)
        
        gp.eval()
        likelihood_.eval()

        with torch.no_grad():
            f_pred = likelihood_(gp(x_tensor))
            y_mean = f_pred.mean.cpu().numpy()
            y_std = f_pred.stddev.cpu().numpy()

        if self.norm_y and y_scaler is not None:
            y_mean = y_mean * y_scaler.scale_ + y_scaler.mean_
            y_std = y_scaler.scale_ * y_std
            
        return y_mean, y_std

    @_preprocessing_eval
    @_bootstrap_eval
    def eval_metamodel_old(self, samples, b_i=0, parallel=False):
        """
        Original evaluation method that evaluates each GP separately, with optional parallelization.
        """
        slurm_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

        samples_sc = self.scale_x(X=samples, transform_obj=self._x_scaler[f"b_{b_i + 1}"])
        x_tensor = self.convert_to_tensor(samples_sc)

        model_dict = self._gp_poly[f"b_{b_i + 1}"]
        mean_pred, std_pred = {}, {}

        for output, values in model_dict.items():
            mean = np.empty((len(samples), len(values)))
            std = np.empty((len(samples), len(values)))
            
            gps_to_eval = [self._gp_poly[f"b_{b_i + 1}"][output][k] for k in values.keys()]
            scalers = [
                self._y_scaler[f"b_{b_i + 1}"][output][k] if self.norm_y else None 
                for k in values.keys()
            ]

            if parallel:
                results = Parallel(n_jobs=slurm_cores, backend="loky", batch_size="auto")(
                    delayed(self._evaluate_single_gp)(gp, x_tensor, scaler)
                    for gp, scaler in zip(gps_to_eval, scalers)
                )
                for idx, (y_mean, y_std) in enumerate(results):
                    mean[:, idx] = y_mean
                    std[:, idx] = y_std
            else:
                for idx, (gp, scaler) in enumerate(zip(gps_to_eval, scalers)):
                    y_mean, y_std = self._evaluate_single_gp(gp, x_tensor, scaler)
                    mean[:, idx] = y_mean
                    std[:, idx] = y_std

            mean_pred[output] = mean
            std_pred[output] = std

        return mean_pred, std_pred
    
    @_preprocessing_eval
    @_bootstrap_eval
    def eval_metamodel(self, samples, b_i=0, parallel=False):
        """
        New evaluation method that evaluates all GPs in a batch simultaneously, leveraging GPyTorch's batch capabilities for maximum speed.
        This method is significantly faster for large numbers of outputs, but assumes that all outputs were trained
        simultaneously with the same kernel and optimizer settings (as is done in batch_adaptive_regression).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samples_sc = self.scale_x(X=samples, transform_obj=self._x_scaler[f"b_{b_i + 1}"])
        N_test = samples_sc.shape[0]
        
        mean_pred = {}
        std_pred = {}

        model_dict = self._gp_poly[f"b_{b_i + 1}"]

        for key, gp in model_dict.items():
            gp = gp.to(device)
            likelihood_ = gp.likelihood.to(device)
            y_scaler = self._y_scaler[f"b_{b_i + 1}"][key]
            
            # Identify how many tasks (M) are in this batch GP
            # by checking the batch shape of the train_targets
            M_tasks = gp.train_targets.shape[0]

            # Expand test inputs to (M, N_test, D)
            x_tensor = self.convert_to_tensor(samples_sc).to(device)
            x_batch = x_tensor.unsqueeze(0).expand(M_tasks, N_test, x_tensor.shape[1])

            gp.eval()
            likelihood_.eval()

            # --- GPYTORCH SPEED OPTIMIZATIONS ENABLED HERE ---
            # --- MEMORY-SAFE CHUNKED EVALUATION ---
            chunk_size = 2000  # Decrease this if you still get OOM errors
            y_mean_list = []
            y_std_list = []

            gp.eval()
            likelihood_.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(35):
                for i in range(0, N_test, chunk_size):
                    # Slice a safe chunk of test points
                    x_chunk = x_batch[:, i:i+chunk_size, :]
                    
                    # Evaluate just this chunk
                    f_pred = likelihood_(gp(x_chunk))
                    
                    # Immediately move results to CPU to free GPU VRAM
                    y_mean_list.append(f_pred.mean.cpu().numpy())
                    y_std_list.append(f_pred.stddev.cpu().numpy())
                    
                    # Force PyTorch to release dead memory graphs
                    del f_pred
                    torch.cuda.empty_cache() 

            # Stitch the chunks back together along the N_test axis (axis=1)
            y_mean_batch = np.concatenate(y_mean_list, axis=1)
            y_std_batch = np.concatenate(y_std_list, axis=1)

            # Transpose back to (N_test, M_tasks) BEFORE scaling
            y_mean = y_mean_batch.T
            y_std = y_std_batch.T

            # Inverse scale the entire matrix at once
            if self.norm_y and y_scaler is not None:
                y_mean = y_scaler.inverse_transform(y_mean)
                # Standard deviation scales linearly
                y_std = y_std * y_scaler.scale_

            mean_pred[key] = y_mean
            std_pred[key] = y_std
            
            # Clean up GPU memory
            gp = gp.cpu()

        return mean_pred, std_pred

