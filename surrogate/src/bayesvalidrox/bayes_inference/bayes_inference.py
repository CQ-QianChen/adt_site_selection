#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform Bayesian inference on trained engine using a PostSampler object.
"""

import copy
import gc
import os
import platform
import warnings

import corner
import h5py
import matplotlib.lines as mlines
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from .mcmc import MCMC
from .rejection_sampler import RejectionSampler

# Load the mplstyle
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


class BayesInference:
    """
    A class to perform Bayesian Analysis.

    Attributes
    ----------
    engine : obj
        A (trained) bvr.Engine object.
    observation : obj
        An object of class bvr.Observation that contains the observation data
        and related uncertainty.
    use_emulator : bool, optional
        Set to True if the emulator/metamodel should be used in the analysis.
        If False, the model is run. The default is True.
    name : str, optional
        The type of analysis, either calibration (`calib`) or validation
        (`valid`). This is mainly used to name the output folder.
        The default is `'calib'`.
    out_names : list, optional
        The list of requested output keys to be used for the analysis.
        If None, all the defined outputs from the engine are used.
        The default is `None`.
    selected_indices : dict, optional
        A dictionary with the selected indices of each model output. The
        default is `None`. If `None`, all measurement points are used in the
        analysis.
    prior_samples : array of shape (n_samples, n_params), optional
        The samples to be used in the analysis. If None, the samples are drawn
        from the ExpDesign given in the Engine.
        The default is `None`.
    n_prior_samples : int, optional
        Number of samples to be used in the analysis. The default is `500000`.
        If prior_samples are given, they are used instead.
    inference_method : str, optional
        The method for approximating the posterior distribution in the Bayesian
        inference step. The default is `'rejection'`, which stands for
        rejection sampling. A Markov Chain Monte Carlo sampler can be simply
        selected by passing `'MCMC'`.
    mcmc_params : dict, optional
        A dictionary with argsuments that can be set for the MCMC class.
        The default is `None`, resulting in the following MCMC settings:
            >>> mcmc_params:{
                'init_samples': None,  # initial samples
                'n_walkers': 100,  # number of walkers (chain)
                'n_steps': 100000,  # number of maximum steps
                'n_burn': 200,  # number of burn-in steps
                'moves': None,  # Moves for the emcee sampler
                'multiprocessing': False,  # multiprocessing
                'verbose': False # verbosity
                }
    bootstrap_method : string, optional
        Method used for perturbation of the observation.
        If 'gaussian' then Gaussian noise is added to perturb observations,
        if 'none' then no bootstrap is applied.
        If set to 'loocv', leave-one-out cross validation is used to generate an
        observation set.
        The default is 'gaussian'.
    n_bootstrap_itrs : int, optional
        Number of bootstrap iteration. The default is `1`. If bootstrap_method is
        `loocv`, this is set to the total length of the observation data set.
    perturbed_data : array of shape (n_bootstrap_itrs, n_obs), optional
        User defined perturbed data. The default is `[]`.
    bootstrap_noise : float, optional
        A noise level to perturb the data set, used if bootstrap_method is set to
        'gaussian'. The default is `0.05`.
    valid_metrics : list, optional
        List of the validation metrics. The following metrics are supported:
        1. KLD : Kullback-Leibler Divergence
        2. inf_entropy: Information entropy
        The default is `[]`.
    plot : bool, optional
        Toggles evaluation plots including posterior predictive plots and plots
        of the model outputs vs the metamodel predictions for the maximum
        a posteriori (defined as `max_a_posteriori`) parameter set. The
        default is `True`.
    max_a_posteriori : str, optional
        Maximum a posteriori. `'mean'` and `'mode'` are available. The default
        is `'mean'`.
    out_dir : str, optional
        The output directory that any generated plots are saved in. The
        default '' leads to the folder
        "Outputs_Bayes_{self.engine.model.name}_{self.name}".
    out_format : str, optional
        Format that the generated plots are stored as. Supports 'pdf' and 'png'.
        The default is 'pdf'.
    """

    def __init__(
        self,
        engine,
        observation,
        use_emulator=True,
        name="Calib",
        out_names=None,
        selected_indices=None,
        prior_samples=None,
        n_prior_samples=100000,
        inference_method="rejection",
        mcmc_params=None,
        bootstrap_method="gaussian",
        n_bootstrap_itrs=1,
        perturbed_data: list = None,
        bootstrap_noise=0.05,
        valid_metrics=None,
        plot=True,
        max_a_posteriori="mean",
        out_dir="",
        out_format="pdf",
    ):

        self.engine = engine
        self.observation = observation
        self.use_emulator = use_emulator
        self.name = name
        self.out_names = engine.out_names if out_names is None else out_names
        self.selected_indices = selected_indices
        self.prior_samples = prior_samples
        self.n_prior_samples = n_prior_samples
        self.inference_method = inference_method
        self.mcmc_params = mcmc_params
        self.perturbed_data = (
            perturbed_data if (perturbed_data is not None) else np.array([])
        )
        self.bootstrap_method = bootstrap_method
        self.n_bootstrap_itrs = n_bootstrap_itrs
        self.bootstrap_noise = bootstrap_noise
        self.valid_metrics = valid_metrics if valid_metrics is not None else []
        self.plot = plot
        self.max_a_posteriori = max_a_posteriori
        self.out_dir = out_dir
        self.out_format = out_format

        # Init of the sampler object
        self.sampler = None

        # Other properties and parameters
        self.posterior_df = None
        self.perturbed_indices = None

        # Prior results
        self.prior_out = None
        self.prior_out_std = None
        self.surr_error = None  # MetaModel rmse, read in prior_eval

        # Validation criteria results
        self.inf_entropy = None
        self.log_bme = None
        self.kld = None

        # Map results
        self.map_orig_model = None
        self.map_metamodel_mean = None
        self.map_metamodel_std = None

        # System settings
        if platform.system() == "Windows" or platform.system() == "Darwin":
            warnings.warn(
                "Performing the inference on windows or MacOS can lead to reduced accuracy!"
            )
            self.dtype = np.longdouble
        else:
            self.dtype = np.float128

    def setup(self):
        """
        This function sets up the inference by checking the inputs and getting
        needed data.
        """
        if self.observation is None:
            raise AttributeError("Cannot run BayesInference, no observation given!")

        if self.name.lower() not in ["valid", "calib"]:
            raise AttributeError(
                "The set inference type is not known! Use either `calib` or `valid`"
            )

        # Create output directory
        if self.out_dir == "":
            self.out_dir = f"Outputs_Bayes_{self.engine.model.name}_{self.name}"
        os.makedirs(self.out_dir, exist_ok=True)

        # If the prior is set by the user, take it, else generate from ExpDes
        if self.prior_samples is None:
            print("Generating prior samples from the Experimental Design.")
            self.prior_samples = self.engine.exp_design.generate_samples(
                self.n_prior_samples, "random"
            )
        else:
            if self.prior_samples.shape[1] != self.engine.meta_model.ndim:
                raise AttributeError(
                    "The given prior_samples do not "
                    "match the expected input dimension."
                )
            self.n_prior_samples = self.prior_samples.shape[0]

        # Read observation data
        self.observation.build()

        # Init selected_indices if not given (per output, all valid indices)
        if self.selected_indices is None:
            self.selected_indices = {}
            for key in self.out_names:
                vals = self.observation.data[key]  # .to_numpy()
                self.selected_indices[key] = np.where(~np.isnan(vals))[0]

        if len(self.perturbed_data) != 0:
            if self.perturbed_data.shape[1] != self.observation.n_values:
                raise ValueError(
                    f"The shape of the perturbed data ({self.perturbed_data.shape[1]}) doesn't "
                    f"fit the given observations ({self.observation.n_values})"
                )
            self.n_bootstrap_itrs = len(self.perturbed_data)
            self.bootstrap_method = "user"

        # Choose number of bootstrap iterations
        if self.bootstrap_method == "loocv":
            if self.n_bootstrap_itrs != self.observation.n_values:
                warnings.warn(
                    "Number of bootstrap iterations is set to the number "
                    "of observation values for LOOCV."
                )
            self.n_bootstrap_itrs = self.observation.n_values

        if self.n_bootstrap_itrs <= 1 or self.bootstrap_method == "none":
            self.n_bootstrap_itrs = 1
            self.bootstrap_method = "none"
            warnings.warn("The inference is performed without bootstrap.")

        # Setup MCMC
        if self.inference_method.lower() == "mcmc":
            if self.mcmc_params is None:
                self.mcmc_params = {}
            par_list = [
                "prior_samples",
                "n_walkers",
                "n_burn",
                "n_steps",
                "moves",
                "multiprocessing",
                "verbose",
            ]
            init_val = [None, 100, 200, 100000, None, False, False]
            for i, _ in enumerate(par_list):
                if par_list[i] not in list(self.mcmc_params.keys()):
                    self.mcmc_params[par_list[i]] = init_val[i]

            self.sampler = MCMC(
                engine=self.engine,
                mcmc_params=self.mcmc_params,
                observation=copy.deepcopy(self.observation),
                out_names=self.out_names,
                selected_indices=self.selected_indices,
                use_emulator=self.use_emulator,
                out_dir=self.out_dir,
            )
        # Rejection sampling
        elif self.inference_method.lower() == "rejection":
            self.sampler = RejectionSampler(
                engine=self.engine,
                observation=copy.deepcopy(self.observation),
                out_names=self.out_names,
                selected_indices=self.selected_indices,
                use_emulator=self.use_emulator,
                out_dir=self.out_dir,
                prior_samples=self.prior_samples,
            )
        else:
            raise AttributeError("The chosen inference method is not available!")

    def run_inference(self):
        """
        Performs Bayesian inference on the given setup.
        The generated posterior samples are stored as
        self.posterior_df.

        """
        # Setup
        self.name = "calib"
        self.setup()

        # Run model or metamodel on the priors
        self.prior_out, self.prior_out_std = self._eval_engine(
            self.prior_samples, model_key="PriorPred"
        )
        self.surr_error = self.get_surr_error()

        # Perturb the data
        if len(self.perturbed_data) == 0:
            self.perturbed_data = self.perturb_data(self.out_names)

        # Calculate likelihood and bme
        log_likes, self.log_bme = self.calculate_loglik_logbme(
            model_evals=self.prior_out,
            rmse=self.surr_error,
        )

        # Average over perturbed results
        self.sampler.log_likes = np.mean(log_likes, axis=1)
        self.sampler.observation = self.observation

        # Run the sampler
        self.posterior_df = self.sampler.run_sampler()

        # Provide posterior's summary
        print("\n")
        print("-" * 15 + "Posterior summary" + "-" * 15)
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        print(self.posterior_df.describe())
        print("-" * 50)

        # Posterior predictive
        self.posterior_predictive(True)

        # Visualization
        if self.plot:
            self.plot_post_params()
            self.plot_max_a_posteriori()
            self.plot_post_predictive()

    def run_validation(self) -> np.ndarray:
        """
        Validate a model on the given samples by calculating the loglikelihood
        and BME.

        Returns
        -------
        log_bme : np.ndarray
            The log-BME calculated on perturbed reference data.

        """
        # Setup
        self.name = "valid"
        self.setup()

        # Run model or metamodel on the priors
        self.prior_out, self.prior_out_std = self._eval_engine(
            self.prior_samples, model_key="PriorPred"
        )
        self.surr_error = self.get_surr_error()

        # Perturb the data
        if len(self.perturbed_data) == 0:
            self.perturbed_data = self.perturb_data(self.out_names)

        # Calculate likelihood and bme
        log_likes, self.log_bme = self.calculate_loglik_logbme(
            model_evals=self.prior_out,
            rmse=self.surr_error,
        )
        self.sampler.log_likes = log_likes

        # Calculate additional validation metrics
        if self.inference_method.lower() != "rejection":
            warnings.warn(
                "Validation metrics are only available for rejection sampling."
            )
        else:
            if "kld" in list(
                map(str.lower, self.valid_metrics)
            ) or "inf_entropy" in list(map(str.lower, self.valid_metrics)):
                self.kld, self.inf_entropy = self.calculate_valid_metrics(
                    log_likes, self.log_bme
                )

        # Visualization
        if self.plot:
            self.plot_logbme()

        return self.log_bme

    def _eval_engine(self, samples, model_key="") -> tuple[dict, dict]:
        """
        Evaluate the engine to run on the full prior.
        Depending on the set option for use_emulator,
        the model or metamodel is run.

        Parameters
        ----------
        samples : 2d np.array
            Array of shape (#samples, #parameters).
        model_key : string, optional
            Key to give to the model run. The default is '

        Returns
        -------
        prior_out : dict
            Model/Metamodel outputs on the prior samples.
        prior_out_std : dict
            Std of metamodel outputs on the prior samples.
            If self.use_emulator is False, returns None.

        """
        # Evaluate the MetaModel
        if self.use_emulator:
            print("Evaluating the metamodel on the prior samples.")
            out, out_std = self.engine.meta_model.eval_metamodel(samples)

        # Evaluate the model
        else:
            print("Evaluating the model on the prior samples.")
            out, _ = self.engine.model.run_model_parallel(
                samples, key_str=model_key + self.name, store_hdf5=False
            )
            out_std = None

        return out, out_std

    def get_surr_error(self) -> dict:
        """
        Obtain the RMSE of the surrogate from the engine.

        Returns
        -------
        surr_error : dict
            RMSE of metamodel if available.
            Otherwise returns None.

        """
        # Not available without metamodel
        if not self.use_emulator:
            return None

        # Return metamodel rmse
        if "rmse" not in self.engine.valid_metrics.keys():
            warnings.warn("The given engine has no rmse value.")
            rmse = None
        else:
            rmse = {}
            for key in self.engine.out_names:
                rmse[key] = self.engine.valid_metrics["rmse"][key][-1]
        return rmse

    def perturb_data(self, output_names) -> dict:
        """
        Perturbs the observation data using the specified method.
        Returns an array with n_bootstrap_itrs rows of perturbed data.
        The first row includes the original observation data.
        Currently includes methods for loocv, gaussian noise and no perturbation.

        If 'loocv' is chosen, a 2d-array will be returned with
        repeated rows and zero diagonal entries.

        Parameters
        ----------
        output_names : list
            The output names.

        Raises
        ------
        AttributeError

        Returns
        -------
        final_data : dict
            Perturbed data set for each key in output_names.
            Shape of np.ndarray for each key: (n_bootstrap, #xvalues in measurement data)

        """
        # Get values in the data
        data = self.observation.data
        obs = np.concatenate([np.atleast_1d(data[key]) for key in output_names]).astype(
            float
        )

        # LOOCV
        if self.bootstrap_method.lower() == "loocv":
            loocv_data = np.repeat(np.atleast_2d(obs), self.n_bootstrap_itrs, axis=0)
            np.fill_diagonal(loocv_data, np.nan)

            # Array of booleans with False=nan entries. To be used later
            self.perturbed_indices = ~np.isnan(loocv_data)

            return loocv_data

        # Noise-based perturbation
        if self.bootstrap_method.lower() == "gaussian":
            # Perturb observation with Gaussian noise, from measurement error
            sigma2 = self.observation.sigma2
            perturbed_data = None
            for key in output_names:
                # Get noise, with additional bootstrap noise level
                noise = np.sqrt(sigma2[key]) * self.bootstrap_noise
                cov = np.diag(noise**2)
                # Generate perturbed samples
                perturbed_samples = np.random.multivariate_normal(
                    data[key], cov, self.n_bootstrap_itrs
                )

                if perturbed_data is None:
                    perturbed_data = perturbed_samples
                else:
                    perturbed_data = np.hstack((perturbed_data, perturbed_samples))
            perturbed_data[0, :] = obs

            return np.array(perturbed_data)

        # No bootstrapping
        if self.bootstrap_method.lower() == "none":
            perturbed = np.array([obs])
            return perturbed

        raise AttributeError(
            f"The chosen bootstrap method {self.bootstrap_method}\
                                 is not available."
        )

    def calculate_loglik_logbme(
        self,
        model_evals,
        rmse,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate log-likelihoods and logbme on the perturbed data.

        Parameters
        ----------
        model_evals : dict
            Model or metamodel outputs as a dictionary.
        rmse : dict
            Estimation of surrogate error via root mean square error.

        Returns
        -------
        log_likelihood : np.ndarray
            The calculated loglikelihoods.
            Size: (n_samples, n_bootstrap_itr).

        log_bme : np.ndarray
            The log bme. This also accounts for metamodel error, if
            self.use_emulator is True. Size: (1,n_bootstrap_itr).

        """
        # Initilize arrays
        log_likelihoods = np.zeros(
            (self.n_prior_samples, self.n_bootstrap_itrs), dtype=np.float16
        )
        log_bme = np.zeros(self.n_bootstrap_itrs)

        # Start the likelihood-BME computations for the perturbed data
        for itr_idx, data in tqdm(
            enumerate(self.perturbed_data),
            total=self.n_bootstrap_itrs,
            desc="BME calculation for perturbed data",
            ascii=True,
        ):
            # Prepare data dataframe
            # Build per-output slices using the actual output names (no x_values assumed)
            # cols = list(self.out_names)  # keep the order
            # number of non-NaN entries per output column
            # Note: only single observation allowed so far
            nobs = [self.observation.data[c].shape[0] for c in self.out_names]

            # cumulative slicing indices across the flattened perturbed vector
            cum = np.cumsum([0] + nobs)
            indices = list(zip(cum[:-1], cum[1:]))

            # map each output -> its slice from this perturbed row
            data_dict = {
                self.out_names[i]: data[j:k] for i, (j, k) in enumerate(indices)
            }
            self.sampler.observation.data = data_dict

            if self.perturbed_indices is not None:
                # map each output -> its valid indices from this perturbed row
                sel_indices = {
                    self.out_names[i]: self.perturbed_indices[itr_idx][j:k]
                    for i, (j, k) in enumerate(indices)
                }

                for key in self.out_names:
                    valid_positions = np.where(sel_indices[key])[0]
                    if isinstance(self.selected_indices, dict):
                        sel_indices[key] = np.intersect1d(
                            valid_positions, self.selected_indices[key]
                        )
                    else:
                        np.intersect1d(valid_positions, self.selected_indices[key])
                self.sampler.selected_indices = sel_indices

            # Calculate loglik and bme
            loglik, logbme = self.sampler.calculate_loglik_logbme(
                model_evals, rmse=rmse
            )
            log_likelihoods[:, itr_idx] = loglik
            log_bme[itr_idx] = logbme

        # Return to default values
        self.sampler.selected_indices = self.selected_indices
        self.sampler.observation = self.observation

        return log_likelihoods, log_bme

    def calculate_valid_metrics(
        self, log_likelihoods, log_bme
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate KLD and information entropy if noted in self.valid_metrics.

        Parameters
        ----------
        log_likelihood : np.array
            Calculated loglikelihoods.
            Size: (n_samples, n_bootstrap_itr).
        log_bme : np.array
            Calculated log bme. This should include for metamodel error, if
            self.use_emulator is True. Size: (1,n_bootstrap_itr).

        Raises
        ------
        AttributeError

        Returns
        -------
        kld : np.ndarray
            Calculated KLD, size: (1,n_bootstrap_itr).
        inf_entropy : np.ndarray
            Calculated information entropy, size: (1,n_bootstrap_itr).

        """

        n_itrs = log_likelihoods.shape[1]
        kld = np.zeros(n_itrs)
        inf_entropy = np.zeros(n_itrs)

        for itr_idx in tqdm(
            range(n_itrs),
            total=n_itrs,
            desc="Calculating the validation metrics",
            ascii=True,
        ):
            # Run rejection sampling for every perturbed likelihood
            self.sampler.log_likes = log_likelihoods[:, itr_idx]
            self.sampler.prior_samples = self.prior_samples
            _ = self.sampler.run_sampler(recalculate_loglik=False)

            # Calculate the validation metrics
            self.sampler.log_bme = log_bme[itr_idx]
            kld_, inf_entropy_ = self.sampler.calculate_valid_metrics(
                self.engine.exp_design, parallel=False
            )
            kld[itr_idx] = kld_
            inf_entropy[itr_idx] = inf_entropy_

            # Clear memory
            gc.collect(generation=2)

        if "kld" not in list(map(str.lower, self.valid_metrics)):
            kld = None
        if "inf_entropy" not in list(map(str.lower, self.valid_metrics)):
            inf_entropy = None
        return kld, inf_entropy

    def posterior_predictive(self, save=False):
        """
        Evaluates the engine on the prior- and posterior predictive samples,
        and stores the results as hdf5 files.

        priorPredictive.hdf5 : Prior predictive samples.
        postPredictive_wo_noise.hdf5 : Posterior predictive samples
            without the additive noise.
        postPredictive.hdf5 : Posterior predictive samples with the
            additive noise.

        Parameters
        ----------
        save : bool, optional
            Toggles storing the posterior predictives as hdf5 files.
            The default is False.

        """
        # Posterior predictive
        post_pred, post_pred_std = self._eval_engine(
            self.posterior_df.values, model_key="PostPred"
        )

        # Add discrepancy from likelihood samples to the current posterior runs
        post_pred_withnoise = copy.deepcopy(post_pred)
        for _, var in enumerate(self.out_names):
            for i in range(len(post_pred[var])):
                pred = post_pred[var][i]

                # Known sigma2s
                clean_sigma2 = self.observation.sigma2[var][
                    ~np.isnan(self.observation.sigma2[var])
                ]
                tot_sigma2 = clean_sigma2[: len(pred)]
                cov = np.diag(tot_sigma2)

                # Add predictive metamodel error/uncertainty
                # Expected value of variance (Assump: i.i.d stds)
                if self.use_emulator:
                    if self.surr_error is not None:
                        std_metamod = self.surr_error[var]
                    else:
                        std_metamod = post_pred_std[var][i]
                    cov += np.diag(std_metamod**2)

                # Sample a multivariate normal distribution with mean of
                # posterior prediction and variance of cov
                post_pred_withnoise[var][i] = np.random.multivariate_normal(
                    pred, cov, 1
                )

        # Store the predictives
        if save:
            x_values = self.engine.exp_design.x_values
            self.write_as_hdf5("priorPredictive.hdf5", self.prior_out, x_values)
            self.write_as_hdf5("postPredictive_wo_noise.hdf5", post_pred, x_values)
            self.write_as_hdf5("postPredictive.hdf5", post_pred_withnoise, x_values)

    def write_as_hdf5(self, name, data, x_values):
        """
        Write given values to an hdf5 file.

        Parameters
        ----------
        name : string
            Filename to write to.
        data : dict
            Data to write out. Is expected to be model or metamodel runs.
        x_values : list
            The x_values that correspond to the written data.

        """
        if data is None:
            raise AttributeError("Given data is empty!")

        # Create hdf5 metadata
        hdf5_file = self.out_dir + "/" + name
        hdf5_exist = os.path.exists(hdf5_file)
        if hdf5_exist:
            os.remove(hdf5_file)
        file = h5py.File(hdf5_file, "a")

        # Store x_values
        file.create_dataset("x_values", data=x_values)

        # Store posterior predictive
        grp_y = file.create_group("EDY/")
        for _, var in enumerate(self.out_names):
            grp_y.create_dataset(var, data=data[var])

    def plot_max_a_posteriori(self):
        """
        Plots the response of the model output against that of the metamodel at
        the maximum a posteriori sample (mean or mode of posterior.)

        """

        meta_model = self.engine.meta_model
        posterior_df = self.posterior_df.values

        # Compute the MAP
        if self.max_a_posteriori.lower() == "mean":
            map_theta = posterior_df.mean(axis=0).reshape((1, meta_model.ndim))
        else:
            map_theta = stats.mode(posterior_df.values, axis=0)[0]
        print("\nPoint estimator:\n", map_theta[0])

        # Run the models for MAP
        # meta_model
        map_metamodel_mean, map_metamodel_std = meta_model.eval_metamodel(map_theta)
        self.map_metamodel_mean = map_metamodel_mean
        self.map_metamodel_std = map_metamodel_std

        # origModel
        map_orig_model, _ = self.engine.model.run_model_parallel(
            map_theta, key_str="MAP" + self.name, store_hdf5=False
        )
        self.map_orig_model = map_orig_model

        # Extract slicing index
        x_values = map_orig_model["x_values"]

        # List of markers and colors
        color = ["k", "b", "g", "r"]
        marker = "x"

        # Create a PdfPages object
        for i, key in enumerate(self.out_names):
            fig = plt.figure()
            y_val = map_orig_model[key][0]
            y_pce_val = map_metamodel_mean[key][0]
            y_pce_val_std = map_metamodel_std[key][0]

            plt.plot(
                x_values,
                y_val,
                color=color[i],
                marker=marker,
                lw=2.0,
                label="$Y_{MAP}^{M}$",
            )

            plt.plot(
                x_values,
                y_pce_val,
                color=color[i],
                lw=2.0,
                marker=marker,
                linestyle="--",
                label="$Y_{MAP}^{PCE}$",
            )
            # Plot the confidence interval
            plt.fill_between(
                x_values,
                y_pce_val - 1.96 * y_pce_val_std,
                y_pce_val + 1.96 * y_pce_val_std,
                color=color[i],
                alpha=0.15,
            )

            # Calculate the adjusted R_squared and RMSE
            r_2 = r2_score(y_pce_val.reshape(-1, 1), y_val.reshape(-1, 1))
            rmse = np.sqrt(mean_squared_error(y_pce_val, y_val))

            plt.ylabel(key)
            plt.xlabel("Time [s]")
            plt.title(f"Model vs MetaModel {key}")

            ax = fig.axes[0]
            fig.canvas.draw()
            ax.text(
                0.8,
                0.13,
                f"RMSE = {rmse:.3f}\n$R^2$ = {r_2:.3f}",
                transform=ax.transAxes,
                color="black",
                bbox={
                    "facecolor": "none",
                    "edgecolor": "black",
                    "boxstyle": "round,pad=1",
                },
            )
            fig.savefig(
                f"./{self.out_dir}/MAP_PCE_vs_Model_{self.name}.{self.out_format}",
                bbox_inches="tight",
            )
            plt.close()

    def plot_post_params(self, corner_title_fmt=".2e"):
        """
        Plots the multivar. posterior parameter distribution.

        Parameters
        ----------
        corner_title_fmt : str, optional
            Title format for the posterior distribution plot with python
            package `corner`. The default is `'.2e'`.

        """
        par_names = self.engine.exp_design.par_names

        # Pot with corner
        fig_posterior = corner.corner(
            self.posterior_df.to_numpy(),
            labels=par_names,
            quantiles=[0.15, 0.5, 0.85],
            show_titles=True,
            title_fmt=corner_title_fmt,
            labelpad=0.2,
            use_math_text=True,
            title_kwargs={"fontsize": 28},
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            smooth=0.5,
            smooth1d=0.5,
        )

        # Loop over axes and set x limits
        axes = np.array(fig_posterior.axes).reshape((len(par_names), len(par_names)))
        for y_i in range(len(par_names)):
            ax = axes[y_i, y_i]
            ax.set_xlim(self.engine.exp_design.bound_tuples[y_i])
            for x_i in range(y_i):
                ax = axes[y_i, x_i]
                ax.set_xlim(self.engine.exp_design.bound_tuples[x_i])

        # Turn off gridlines
        for ax in fig_posterior.axes:
            ax.grid(False)

        plotname = f"/Posterior_Dist_{self.engine.model.name}"
        if self.use_emulator:
            plotname += "_emulator"

        fig_posterior.set_size_inches((24, 16))
        fig_posterior.savefig(
            f"./{self.out_dir}{plotname}.{self.out_format}", bbox_inches="tight"
        )

        plt.close()

    def plot_logbme(self):
        """
        Plots the log_BME if bootstrap is active.

        """

        # Computing the TOM performance
        log_bme_tom = stats.chi2.rvs(
            self.observation.n_values, size=self.log_bme.shape[0]
        )

        _, ax = plt.subplots()
        sns.kdeplot(log_bme_tom, ax=ax, color="green", shade=True)
        sns.kdeplot(self.log_bme, ax=ax, color="blue", shade=True, label="Model BME")

        ax.set_xlabel("log$_{10}$(BME)")
        ax.set_ylabel("Probability density")

        legend_elements = [
            Patch(facecolor="green", edgecolor="green", label="TOM BME"),
            Patch(facecolor="blue", edgecolor="blue", label="Model BME"),
        ]
        ax.legend(handles=legend_elements)

        plotname = f"/BME_hist_{self.engine.model.name}"
        if self.use_emulator:
            plotname += "_emulator"

        plt.savefig(
            f"./{self.out_dir}{plotname}.{self.out_format}", bbox_inches="tight"
        )

        plt.close()

    def plot_post_predictive(self):
        """
        Plots the posterior predictives against the observation data.

        """

        model = self.engine.model

        # Calculate measurement error for the plots
        obs = self.observation
        measurement_error = {k: np.sqrt(obs.sigma2[k]) for k in obs.sigma2.keys()}

        # Plot the posterior predictive
        for _, out_name in enumerate(self.out_names):
            fig, ax = plt.subplots()
            with sns.axes_style("ticks"):
                x_key = "x_values"

                # --- Read prior and posterior predictive ---
                if (
                    self.inference_method == "rejection"
                    and self.name.lower() == "calib"
                ):
                    #  --- Prior ---
                    # Load posterior predictive
                    f = h5py.File(f"{self.out_dir}/priorPredictive.hdf5", "r+")
                    x_coords = np.array(f["x_values"])
                    outputs = np.array(f[f"EDY/{out_name}"])[
                        : self.n_prior_samples
                    ].flatten("F")
                    n_prior = int(outputs.shape[0] / x_coords.shape[0])
                    x_values = np.repeat(x_coords, n_prior)

                    prior_pred_df = {
                        x_key: x_values,
                        out_name: outputs,
                    }
                    prior_pred_df = pd.DataFrame(prior_pred_df)

                    tags_post = ["prior"] * len(prior_pred_df)
                    prior_pred_df.insert(
                        len(prior_pred_df.columns), "Tags", tags_post, True
                    )
                    f.close()

                    # --- Posterior ---
                    f = h5py.File(f"{self.out_dir}/postPredictive.hdf5", "r+")
                    x_values = np.repeat(
                        x_coords, np.array(f[f"EDY/{out_name}"]).shape[0]
                    )

                    post_pred_df = {
                        x_key: x_values,
                        out_name: np.array(f[f"EDY/{out_name}"]).flatten("F"),
                    }
                    post_pred_df = pd.DataFrame(post_pred_df)

                    tags_post = ["posterior"] * len(post_pred_df)
                    post_pred_df.insert(
                        len(post_pred_df.columns), "Tags", tags_post, True
                    )
                    f.close()
                    # Concatenate two dataframes based on x_values
                    frames = [prior_pred_df, post_pred_df]
                    all_pred_df = pd.concat(frames)

                    # --- Plot posterior predictive ---
                    sns.violinplot(x=x_key,
                                   y=out_name,
                                   data=all_pred_df,
                                   hue="Tags",
                                   legend=False,
                                   ax=ax,
                                   split=True,
                                   inner=None,
                                   color=".8",
)

                    # --- Plot Data ---
                    # Find the x,y coordinates for each point
                    x_coords = np.arange(x_coords.shape[0])
                    obs_data = self.observation.data  # .round({x_key: 6})
                    sns.pointplot(
                        x=x_key,
                        y=out_name,
                        color="g",
                        markers="x",
                        linestyles="",
                        capsize=16,
                        data=obs_data,
                        ax=ax,
                    )

                    ax.errorbar(
                        x_coords,
                        obs_data[out_name],
                        yerr=1.96 * measurement_error[out_name],
                        ecolor="g",
                        fmt=" ",
                        zorder=-1,
                    )

                    # Add labels to the legend
                    handles, labels = ax.get_legend_handles_labels()
                    labels.append("Data")

                    data_marker = mlines.Line2D(
                        [],
                        [],
                        color="lime",
                        marker="+",
                        linestyle="None",
                        markersize=10,
                    )
                    handles.append(data_marker)

                    # Add legend
                    ax.legend(
                        handles=handles,
                        labels=labels,
                        loc="best",
                        fontsize="large",
                        frameon=True,
                    )

                else:
                    # Load posterior predictive
                    f = h5py.File(f"{self.out_dir}/postPredictive.hdf5", "r+")
                    x_coords = np.array(f["x_values"])
                    mu_ = np.mean(np.array(f[f"EDY/{out_name}"]), axis=0)
                    std = np.std(np.array(f[f"EDY/{out_name}"]), axis=0)

                    # --- Plot posterior predictive ---
                    plt.plot(
                        x_coords,
                        mu_,
                        marker="o",
                        color="b",
                        label="Mean Post. Predictive",
                    )
                    plt.fill_between(
                        x_coords,
                        mu_ - 1.96 * std,
                        mu_ + 1.96 * std,
                        color="b",
                        alpha=0.15,
                    )

                    # --- Plot Data ---
                    ax.plot(
                        x_coords,
                        self.observation.data[out_name],
                        "ko",
                        label="data",
                        markeredgecolor="w",
                    )

                    # --- Plot ExpDesign ---
                    for output in self.engine.exp_design.y[out_name]:
                        plt.plot(x_coords, output, color="grey", alpha=0.15)

                    # Add labels for axes
                    plt.xlabel("Time [s]")
                    plt.ylabel(out_name)

                    # Add labels to the legend
                    handles, labels = ax.get_legend_handles_labels()

                    patch = Patch(color="b", alpha=0.15)
                    handles.insert(1, patch)
                    labels.insert(1, "95 $\\%$ CI")

                    # Add legend
                    ax.legend(handles=handles, labels=labels, loc="best", frameon=True)

                # Save figure in pdf format
                plotname = f"/Post_Prior_Perd_{model.name}"
                if self.use_emulator:
                    plotname += "_emulator"
                fig.savefig(
                    f"./{self.out_dir}{plotname}_{out_name}.{self.out_format}",
                    bbox_inches="tight",
                )
        plt.clf()
