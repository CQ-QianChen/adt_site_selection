# -*- coding: utf-8 -*-
"""
Engine to train the surrogate

"""
from copy import deepcopy
import os
import warnings
import joblib
import numpy as np
from sklearn.metrics import r2_score

from ..bayes_inference.rejection_sampler import RejectionSampler
from .sequential_design import SequentialDesign
from .supplementary import root_mean_squared_error
from .meta_model import MetaModel as MM


class Engine:
    """
    Engine

    This class is responsible for collecting and managing the experimental
    design, the model and the metamodel for training and evaluations.

    Attributes
    ----------
    meta_model : obj
        A bvr.MetaModel object. If no MetaModel should be trained and used, set
        this to None.
    model : obj
        A model interface of type bvr.PyLinkForwardModel that can be run.
    exp_design : obj
        The experimental design that will be used to sample from the input
        space.
    discrepancy : obj, optional
        A bvr.Discrepancy object that describes the model uncertainty, i.e. the diagonal entries
        of the variance matrix for a multivariate normal likelihood. This is used
        during active learning. The default is None.

    """

    def __init__(self, meta_model, model, exp_design, observation=None):
        self.meta_model = meta_model
        self.model = model
        self.exp_design = exp_design
        self.observation = observation
        self.parallel = False
        self.trained = False

        # Init general parameters
        self.out_names = None
        self.par_names = None
        self.emulator = False
        self.verbose = False

        # Init AL parameters
        self.has_mc = -1
        self.has_observation = -1
        self.valid_metrics = {}
        self.bayes_metrics = {}
        self.n_samples = []
        self.seq_des = None

    def start_engine(self) -> None:
        """
        Do all the preparations that need to be run before the actual training

        Returns
        -------
        None

        """
        self.out_names = self.model.output.names
        self.exp_design.out_names = self.out_names

        if isinstance(self.meta_model, MM):
            self.emulator = True
            self.meta_model.out_names = self.out_names
            self.meta_model.add_parnames()
            if self.verbose:
                print("MetaModel has been given, `emulator` will be set to `True`")
        else:
            self.emulator = False
            if self.verbose:
                print("MetaModel has not been given, `emulator` will be set to `False`")

    def train_normal(self, parallel=False, verbose=False, save=False) -> None:
        """
        Trains surrogate on static samples only.
        Samples are taken from the experimental design and the specified
        model is run on them.
        Alternatively the samples can be read in from a provided hdf5 file.
        save: bool determines whether the trained surrogate and the hdf5 file should be saved

        Returns
        -------
        None

        """
        self.verbose = verbose
        self.start_engine()

        exp_design = self.exp_design
        meta_model = self.meta_model

        # Prepare X samples
        max_deg = np.max(meta_model.pce_deg) if self.emulator else 1
        exp_design.generate_ed(max_deg=max_deg)

        # Run simulations at X
        if not hasattr(exp_design, "y") or exp_design.y is None:
            print("\n Now the forward model needs to be run!\n")

            self.model.delete_hdf5(f"ExpDesign_{self.model.name}.hdf5")

            y_train, new_x_train = self.model.run_model_parallel(
                exp_design.x, mp=parallel, store_hdf5=save
            )
            exp_design.x = new_x_train
            exp_design.y = y_train
        else:
            # Check if a dict has been passed.
            if not isinstance(exp_design.y, dict):
                raise TypeError(
                    "Please provide either a dictionary or a hdf5"
                    "file to exp_design.hdf5_file argument."
                )

        # Separate output dict and x-values
        if "x_values" in exp_design.y:
            exp_design.x_values = exp_design.y["x_values"]
            del exp_design.y["x_values"]
        else:
            if self.verbose:
                print(
                    "No x_values are given, this might lead to issues during PostProcessing"
                )

        # Fit the surrogate
        if self.emulator:
            meta_model.fit(
                exp_design.x, exp_design.y, parallel=parallel, verbose=verbose
            )

        # Save what there is to save
        if save:
            # Save surrogate
            if not os.path.exists("surrogates/"):
                os.makedirs("surrogates/")
            with open(f"surrogates/surrogate_{self.model.name}.pk1", "wb") as output:
                joblib.dump(meta_model, output, 2)

            # Zip the model run directories
            if (
                self.model.link_type.lower() == "pylink"
                and self.exp_design.sampling_method.lower() != "user"
            ):
                self.model.zip_subdirs(self.model.name, f"{self.model.name}_")

        # Set that training was done
        self.trained = True

        # Validation
        if self.emulator:
            self.validate(verbose=verbose)

    # -------------------------------------------------------------------------
    def eval_metamodel(
        self,
        samples=None,
        nsamples=None,
        sampling_method="random",
        return_samples=False,
        parallel=False,
    ):
        """
        Evaluates metamodel at the requested samples. One can also generate
        nsamples.

        Parameters
        ----------
        samples : array of shape (n_samples, n_params), optional
            Samples to evaluate metamodel at. The default is None.
        nsamples : int, optional
            Number of samples to generate, if no `samples` is provided. The
            default is None.
        sampling_method : str, optional
            Type of sampling, if no `samples` is provided. The default is
            'random'.
        return_samples : bool, optional
            Retun samples, if no `samples` is provided. The default is False.
        parallel : bool, optional
            Set to True if the evaluations should be done in parallel.
            The default is False.

        Returns
        -------
        mean_pred : dict
            Mean of the predictions.
        std_pred : dict
            Standard deviatioon of the predictions.
        """
        if samples is None and nsamples is None:
            return None

        # Generate samples
        if samples is None:
            samples = self.exp_design.generate_samples(nsamples, sampling_method)

        # Evaluate Model or MetaModel
        if self.emulator:
            # MetaModel does internal transformation to other space
            mean_pred, std_pred = self.meta_model.eval_metamodel(samples, parallel=parallel)
        else:
            mean_pred, _ = self.model.run_model_parallel(samples, mp=parallel)

        if return_samples:
            if self.emulator:
                return mean_pred, std_pred, samples
            return mean_pred, None, samples
        if self.emulator:
            return mean_pred, std_pred
        return mean_pred, None

    # -------------------------------------------------------------------------
    def add_to_valid(self, name: str, values: dict, v_type: str, storage=None):
        """
        Add the validation output to the valid_metrics.

        Parameters
        ----------
        name : str
            Name of validation criteria
        values : dict
            Values evaluated for the criteria, one value per output.
        v_type : str
            Type of validation criteria, supports 'valid' and 'bayes'.
        storage : list of dicts, optional
            Dictionaries to store the values in, if not to be stored
            in self.valid_metrics/bayes_metrics
        """
        v_store = self.valid_metrics
        b_store = self.bayes_metrics
        if storage is not None:
            v_store = storage[0]
            b_store = storage[1]
        if v_type not in ["valid", "bayes"]:
            raise AttributeError("Given validation type is not valid.")
        if not isinstance(values, dict) and v_type == "valid":
            raise AttributeError("Type 'valid' expects a dictionary.")
        if isinstance(values, dict) and v_type == "bayes":
            raise AttributeError("Type 'bayes' expects a list or value.")

        if v_type == "valid":
            if not name in v_store:
                v_store[name] = {}
                for out in self.out_names:
                    v_store[name][out] = []
            for out in self.out_names:
                v_store[name][out].append(values[out])
        elif v_type == "bayes":
            if not name in b_store:
                b_store[name] = []
            b_store[name].append(values)
        if storage is not None:
            return v_store, b_store
        return None

    def validate(self, store=True, verbose=False) -> None:
        """
        Evaluate the metamodel for validation.
        Two types of validation methods are considered:
        - Prediction validation: RMSE, MSE, R2, mean and stdev errors
        - Bayesian validation: BME, DKL, InfEntropy

        Parameters
        ----------
        store : bool, optional
            Validation results are stored in internal dictionary if
            set to True. If set to False, they are collected and returned.
            The default is 'True'.
        verbose : bool, optional
            Verbosity of this functin. If set to True, will print the
            calculated metrics.
            The default is 'False'.

        """
        if verbose:
            print("")
            print("Running internal metamodel validation")
            print("")

        storage = None
        if not store:
            storage = [{}, {}]
        else:
            # Store number of training points
            self.n_samples.append(self.exp_design.x.shape[0])

        # ---- Prediction validation ----
        # Compare with validation model runs
        rmse = None
        if self.exp_design.y_valid is not None:
            rmse, mse, r2, _ = self._valid_error()
            storage = self.add_to_valid("rmse", rmse, "valid", storage)
            storage = self.add_to_valid("mse", mse, "valid", storage)
            storage = self.add_to_valid("r2", r2, "valid", storage)
        else:
            warnings.warn(
                "No validation samples given, some validation criteria are disabled."
            )

        # Check for mc-reference
        # Get issues from these comparisons, thus doubled it here
        if self.has_mc == -1 or self.model.mc_reference != {}:
            # Try to read it in
            self.has_mc = 1
            try:
                self.model.read_mcref()
            except AttributeError:
                self.has_mc = 0
                warnings.warn(
                    "No mc-reference is given, some validation criteria are disabled."
                )

        # Check the convergence of the Mean & Std
        if self.has_mc:
            err_mean, err_std = self._error_mean_std()
            storage = self.add_to_valid("mean_err", err_mean, "valid", storage)
            storage = self.add_to_valid("std_err", err_std, "valid", storage)

        # Calculate modified loo error
        if self.meta_model.loocv_score_dict is not None:
            scores_all, var_expdes_y_all = [], []
            mod_loo = {}
            for out in self.out_names:
                y = self.exp_design.y[out]
                scores = list(self.meta_model.loocv_score_dict["b_1"][out].values())
                scores_all.append(scores)
                if self.meta_model.dim_red_method.lower() == "pca":
                    pca = self.meta_model.pca["b_1"][out]
                    components = pca.transform(y)
                    var_expdes_y = np.var(components, axis=0)
                else:
                    var_expdes_y = np.var(y, axis=0)
                var_expdes_y_all.append(var_expdes_y)
                mod_loo[out] = np.average(
                    [1 - score for score in scores], weights=var_expdes_y
                )
            storage = self.add_to_valid("mod_loo", mod_loo, "valid", storage)

        # ---- Bayesian validation ----
        # Read observations or MC-reference
        if self.has_observation == -1:
            if self.observation is not None:
                self.observation.build()
                self.has_observation = 1
            else:
                self.has_observation = 0
                warnings.warn(
                    "No observation given, some validation criteria are disabled."
                )

        # Check if discrepancy is provided
        if self.has_observation:
            out = self._bme_calculator(rmse)
            storage = self.add_to_valid("log_bme", out[0], "bayes", storage)
            storage = self.add_to_valid("kld", out[1], "bayes", storage)
            storage = self.add_to_valid("inf_ent", out[2], "bayes", storage)

        # Show statistics
        if verbose:
            print("")
            print("-------------------")
            print("Updated validation outputs:")
            for key, valid in self.valid_metrics.items():
                valid_str = ""
                for out in self.out_names:
                    if valid_str == "":
                        valid_str = f"{out}: {valid[out][-1]}"
                    else:
                        valid_str = valid_str + f", {out}: {valid[out][-1]}"
                print(f"{key}: {valid_str}")
            for key, bayes in self.bayes_metrics.items():
                print(f"{key}: {bayes[-1]}")
        return storage

    def train_sequential(self, parallel=False, verbose=False) -> None:
        """
        Train the surrogate in a sequential manner.
        First build and train evereything on the static samples, then iterate
        choosing more samples and refitting the surrogate on them.

        Parameters
        ----------
        parallel : bool, optional
            Toggles parallelization in the MetaModel training.
            The default is False
        verbose : bool, optional
            Toggles verbose outputs during training.
            The default is False.

        Returns
        -------
        None

        """

        # ---------- Initial self.meta_model ----------
        if not self.trained:
            self.start_engine()
            self.train_normal(parallel=parallel, verbose=verbose)

        # Setup the Sequential Design object ---------------------------------
        self.seq_des = SequentialDesign(
            self.meta_model,
            self.exp_design,
            self.observation,
            out_names=self.out_names,
            parallel=parallel,
            verbose=verbose,
        )

        # Static expdesign parameters for sequential design
        init_n_samples = self.exp_design.n_init_samples
        n_itrs = self.exp_design.n_max_samples - init_n_samples

        # ------- Start Sequential Experimental Design -------
        x = self.exp_design.x
        y = self.exp_design.y
        curr_n_samples = init_n_samples

        for itr_no in range(1, n_itrs + 1):
            print(f"\n>>>> Iteration number {itr_no} <<<<")

            # Save previous meta_model for adaptive tradeoff scheme
            prev_mm = deepcopy(self.meta_model)

            # Choose new sample
            x_new = self.seq_des.choose_next_sample()

            # Evaluate the model at the new sample
            y_new, _ = self.model.run_model_parallel(x_new, prev_run_no=curr_n_samples)

            # Update experimental design
            curr_n_samples += x_new.shape[0]
            x = np.vstack((x, x_new))
            self.exp_design.x = x

            for out_name in self.out_names:
                y_full = np.vstack((y[out_name], y_new[out_name]))
                self.exp_design.y[out_name] = y_full
            y = self.exp_design.y

            # Train the surrogate model for new exp_design
            self.train_normal(parallel=parallel)

            # Save the meta_model prediction before updating
            self.seq_des.prev_mm = prev_mm

            # Clean up
            print()
            print("-" * 50)
            print()

    # -------------------------------------------------------------------------
    def _bme_calculator(self, rmse=None):
        """
        This function computes the Bayesian model evidence (BME) via Monte
        Carlo integration.

        Parameters
        ----------
        rmse : dict of floats, optional
            RMSE values for each output-key. The dafault is None.

        Returns
        -------
        (log_bme, kld, x_post, likelihoods, dist_hellinger)

        """
        # Initializations
        sampling_method = "random"
        mc_size = 10000
        ess = 0
        # Estimation of the integral via Monte Varlo integration
        while (ess > mc_size) or (ess < 1):

            # Generate samples for Monte Carlo simulation
            x_mc = self.exp_design.generate_samples(mc_size, sampling_method)

            # Monte Carlo simulation for the candidate design
            y_mc, std_mc = self.meta_model.eval_metamodel(x_mc)

            # Rejection step
            sampler = RejectionSampler(
                prior_samples=x_mc,
                use_emulator=False,
                out_names=self.out_names,
                observation=self.observation,
            )
            sampler.posterior = sampler.run_sampler(
                outputs=y_mc,
                surr_error=rmse,
                std_outputs=std_mc,
                consider_samplesize=True,
                recalculate_loglik=True,
            )

            # Enlarge sample size if it doesn't fulfill the criteria
            ess = sampler.ess
            if (ess > mc_size) or (ess < 1):
                mc_size *= 10
                print(f"ess={ess}, increasing the MC size to {mc_size}.")

        # Validation metrics
        kld, inf_ent = sampler.calculate_valid_metrics(self.exp_design)

        return (
            sampler.log_bme,
            kld,
            inf_ent,
            sampler.posterior,
            sampler.likelihoods,
        )

    # -------------------------------------------------------------------------
    def _valid_error(self):
        """
        Evaluate the meta_model on the validation samples and calculate the
        error against the corresponding model runs

        Returns
        -------
        rmse : dict
            RMSE for each validation run.
        mse : dict
            MSE for each validation run.
        r2 : dict
            R2 for each validation run.
        valid_error : dict
            Normed (?)RMSE for each validation run.

        """
        # Obtain model and surrogate outputs
        valid_model_runs = self.exp_design.y_valid
        valid_metamod_runs, _ = self.meta_model.eval_metamodel(self.exp_design.x_valid)

        # Loop over the keys and compute RMSE error.
        rmse = {}
        mse = {}
        r2 = {}
        valid_error = {}
        for key in self.out_names:
            rmse[key] = root_mean_squared_error(
                valid_model_runs[key], valid_metamod_runs[key]
            )
            mse[key] = np.power(rmse[key], 2)
            r2[key] = r2_score(
                np.array(valid_metamod_runs[key]).reshape(-1, 1),
                np.array(valid_model_runs[key]).reshape(-1, 1),
            )

            # Validation error
            valid_error[key] = np.power(rmse[key], 2)
            valid_error[key] /= np.var(valid_model_runs[key], ddof=1, axis=0)

        return rmse, mse, r2, valid_error

    # -------------------------------------------------------------------------
    def _error_mean_std(self):
        """
        Calculates the error in the overall mean and std approximation of the
        surrogate against the mc-reference provided to the model.

        Returns
        -------
        rmse_mean : float
            RMSE of the means
        rmse_std : float
            RMSE of the standard deviations

        """
        if self.model.mc_reference == {}:
            raise AttributeError(
                "Model.mc_reference needs to be given to calculate the surrogate error!"
            )

        # Compute the mean and std based on the meta_model
        means, stds = self.meta_model.calculate_moments()

        rmse_mean, rmse_std = {}, {}
        # Compute the root mean squared error between metamodel outputs and mc ref
        for output in self.out_names:
            rmse_mean[output] = root_mean_squared_error(
                self.model.mc_reference["mean"][output], means[output]
            )
            rmse_std[output] = root_mean_squared_error(
                self.model.mc_reference["std"][output], stds[output]
            )

        return rmse_mean, rmse_std
