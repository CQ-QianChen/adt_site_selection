#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template and parent class for classes that generate samples from the posterior.
"""

import gc
import platform
import warnings
import numpy as np
from scipy import stats
import pandas as pd


class PostSampler:
    """
    Template class for generating posterior samples.
    This class describes all the properties and functions that are needed to
    interface with the class BayesInference.

    Attributes
    ----------
    engine : object, optional
        Trained bvr.Engine object. The default is None.
    observation : obj, optional
        An object of class bvr.Observation that contains the observation data
        and related uncertainty.
        The default is None.
    out_names : list, optional
        The list of requested output keys to be used for the analysis.
        The default is `None`. If None, all the defined outputs from the engine
        are used.
    selected_indices : dict, optional
        A dictionary with the selected indices of each model output. The
        default is `None`. If `None`, all measurement points are used in the
        analysis.
    use_emulator : bool
        Set to True if the emulator/metamodel should be used in the analysis.
        If False, the model is run.
    out_dir : string, optional
        The output directory. The default is ''.
    """

    def __init__(
        self,
        engine=None,
        observation=None,
        out_names=None,
        selected_indices=None,
        use_emulator=False,
        out_dir="",
    ):
        # Assign properties - objects
        self.engine = engine
        self.observation = observation

        # Assign properties - settings
        self.selected_indices = selected_indices
        self.out_names = out_names
        self.use_emulator = use_emulator
        self.out_dir = out_dir

        # System settings
        self.dtype = None
        if platform.system() == "Windows" or platform.system() == "Darwin":
            warnings.warn(
                "Performing the inference on windows or MacOS can lead to reduced accuracy!"
            )
            self.dtype = np.longdouble
        else:
            self.dtype = np.float128

    def run_sampler(self) -> pd.DataFrame:
        """
        Performs sampling to update the prior distribution on the
        input parameters.

        Returns
        -------
        posterior : pd.DataFrame
            Posterior samples of the input parameters.

        """
        return None

    def check_observation(self) -> None:
        """
        Checks for match between observation keys and self.out_names
        """
        obs = self.observation
        for out in self.out_names:
            if out not in obs.out_names:
                raise KeyError(
                    f"Observation is missing output '{out}'. "
                    f"Available keys: {obs.out_names}"
                )

    def check_indices(self) -> None:
        """
        Checks for correctly defined indices
        """
        if self.selected_indices is not None:
            if not isinstance(self.selected_indices, (list, dict)):
                raise TypeError(
                    "The selected indices should be given as a list or a dictionary."
                )
            if isinstance(self.selected_indices, dict):
                missing = [
                    key for key in self.out_names if key not in self.selected_indices
                ]
                if missing:
                    raise ValueError(f"Selected indices missing for keys {missing}.")

    # -------------------------------------------------------------------------
    def normpdf(self, outputs, std_outputs=None, rmse=None, norm=True) -> np.ndarray:
        """
        Calculates the likelihood of simulation outputs compared with
        observation data.

        Parameters
        ----------
        outputs : dict
            The metamodel outputs as an array of shape
            (n_samples, n_measurement) for each model output.
        std_outputs : dict of 2d np arrays, optional
            Standard deviation (uncertainty) associated to the output.
            The default is None.
        rmse : dict, optional
            A dictionary containing the root mean squared error as array of
            shape (n_samples, n_measurement) for each model output. The default
            is None.
        norm : bool, optional
            Toggles if the full or reduced version of the normpdf is used.
            If set to False, the normalization with the covariance is removed.
            The default is True.

        Returns
        -------
        loglik : np.ndarray
            Log-likelihoods. Shape: (n_samples)

        """
        # Input checks
        self.check_observation()
        self.check_indices()
        if rmse is None and norm and self.use_emulator:
            warnings.warn("No estimation of surrogate-uncertainty used!")

        nsamples, nout = outputs[self.out_names[0]].shape
        loglik = np.zeros((nsamples,))

        # Loop over the output keys
        for out in self.out_names:
            # Select the data points to compare
            indices = (
                list(range(nout))
                if self.selected_indices is None
                else self.selected_indices
            )
            if isinstance(indices, dict):
                indices = indices[out]

            # Choose values accordingly
            output = outputs[out][:, indices]
            data = self.observation.data[out][indices]
            tot_sigma2s = np.array(self.observation.sigma2[out][indices], dtype=float)

            if not norm:
                # This returns '0' if covmatrix is 0 - check with bmc theory
                # Use only the exponential part of the normpdf
                for p_id, point in enumerate(output):
                    # Add stdev to covmatrix - assumes Gaussian uncertainty
                    covmatrix = np.diag(tot_sigma2s)
                    if std_outputs is not None:
                        covmatrix += np.diag(std_outputs[out][p_id, indices])

                    # Loglik calculation
                    a = np.atleast_2d(data - point)
                    b = np.linalg.pinv(covmatrix)
                    loglik[p_id] += np.ravel(-0.5 * (np.matmul(a, np.matmul(b, a.T))))[0]

            else:
                # This returns -inf if cov = 0, check with bayesinf theory!
                # RMSE as global estimate of surrogate uncertainty
                if rmse is not None:
                    tot_sigma2s += rmse[out][indices] ** 2

                # Set up Covariance Matrix
                covmatrix = np.diag(tot_sigma2s)

                # Use given total_sigma2
                loglik += stats.multivariate_normal.logpdf(
                    output,
                    data,
                    covmatrix,
                    allow_singular=True,
                )

        return loglik

    # -------------------------------------------------------------------------
    def corr_factor_bme(
        self, model_outputs, metamod_outputs, log_bme, norm=True
    ) -> np.ndarray:
        """
        Calculates the correction factor for BMEs.
        This calculation is based on equation 2.30 in the PhD thesis
        of Farid Mohammadi.

        Parameters
        ----------
        model_outputs : dict
            Model outputs on X.
        metamod_outputs : dict
            MetaModel outputs on X.
        log_bme : np.ndarray
            The log_BME obtained from the estimated likelihoods
        norm : bool, optional
            Toggles if the full or reduced version of the normpdf is used.
            If set to False, the normalization with the covariance is removed.
            The default is True.

        Returns
        -------
        np.log(weights) : np.ndarray
            Correction factors for the bme

        """
        # Input checks
        self.check_observation()
        self.check_indices()

        # Check log_bme
        nsamples, nout = model_outputs[self.out_names[0]].shape
        if norm is True:
            if log_bme is None:
                raise AttributeError("No log_bme given for the correction factor.")
            log_bme = np.array(log_bme)
            if len(log_bme.shape) > 1:
                raise AttributeError("Given log_bme has too many dimensions.")

        # Loop over the outputs
        loglik_data = np.zeros((nsamples))
        loglik_model = np.zeros((nsamples))
        for _, out in enumerate(self.out_names):
            # Select the data points to compare
            indices = (
                list(range(nout))
                if self.selected_indices is None
                else self.selected_indices
            )
            if isinstance(indices, dict):
                indices = indices[out]

            # Choose values accordingly
            data = self.observation.data[out]
            tot_sigma2s = self.observation.sigma2[out]
            covmatrix_data = np.diag(tot_sigma2s)

            if norm:
                for i in range(nsamples):
                    # Calculate covMatrix with the surrogate error
                    covmatrix = np.eye(len(model_outputs[out][i])) * 1 / (2 * np.pi)
                    covmatrix = np.diag(covmatrix[indices, indices])

                    covmatrix_data_diag = np.diag(covmatrix_data[indices, indices])

                    # Compute likelilhood output vs data
                    loglik_data[i] += stats.multivariate_normal.logpdf(
                        metamod_outputs[out][i][indices],
                        data[indices],
                        covmatrix_data_diag,
                    )

                    # Compute likelilhood output vs surrogate
                    loglik_model[i] += stats.multivariate_normal.logpdf(
                        metamod_outputs[out][i][indices],
                        model_outputs[out][i][indices],
                        covmatrix,
                    )
            else:
                error = (
                    metamod_outputs[out][:, indices] - model_outputs[out][:, indices]
                )
                for i in range(nsamples):
                    a = np.atleast_2d(error[i, :])
                    b = np.linalg.pinv(np.diag(error[i, :]))
                    loglik_model[i] += -0.5 * (np.matmul(a, np.matmul(b, a.T)))[0, 0]

        # Calculate log weights
        weights = loglik_model
        if norm:
            weights = np.mean(np.exp(loglik_model + loglik_data - log_bme))
            weights = np.log(weights)
        else:
            weights = np.sum(weights)

        return weights

    def calculate_loglik_logbme(
        self, model_evals, rmse=None, std_outputs=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate log-likelihoods and logbme on the perturbed data.
        This function assumes everything as Gaussian.

        Parameters
        ----------
        model_evals : dict
            Model or metamodel outputs as a dictionary.
        rmse : dict, optional
            A dictionary containing the root mean squared error as array of
            shape (n_samples, n_measurement) for each model output. The default
            is None.
        std_outputs : dict of 2d np arrays, optional
            Standard deviation (uncertainty) associated to the output.
            The default is None.

        Returns
        -------
        log_likelihood : np.ndarray
            The calculated loglikelihoods.
            Size: (n_samples, n_bootstrap_itr).

        log_bme : np.ndarray
            The log bme. This also accounts for metamodel error, if
            self.use_emulator is True. Size: (1,n_bootstrap_itr).

        """
        # Log likelihood
        log_likelihoods = self.normpdf(model_evals, rmse=rmse, std_outputs=std_outputs)

        # Calculate logbme
        log_bme = np.log(np.nanmean(np.exp(log_likelihoods, dtype=self.dtype)))

        # BME correction when using Emulator - use training data as validation data
        bme_corr = 0
        if self.use_emulator:
            metamod_outputs, _ = self.engine.eval_metamodel(self.engine.exp_design.x)
            bme_corr = self.corr_factor_bme(
                self.engine.exp_design.y,
                metamod_outputs,
                log_bme,
            )
        log_bme += bme_corr

        # Clear memory
        gc.collect(generation=2)
        return log_likelihoods, log_bme
