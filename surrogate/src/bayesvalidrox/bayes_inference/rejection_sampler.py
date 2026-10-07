#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostSampler for rejection sampling.
"""

import warnings
import multiprocessing
import numpy as np
import pandas as pd

from .post_sampler import PostSampler


class RejectionSampler(PostSampler):
    """
    A class for generating posterior samples via rejection sampling.

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
    prior_samples : np.ndarray, optional
        Prior samples to be used in the rejection sampling.
    """

    def __init__(
        self,
        engine=None,
        observation=None,
        out_names=None,
        selected_indices=None,
        use_emulator=True,
        out_dir="",
        prior_samples=None,
    ):
        # Parent init
        super().__init__(
            engine=engine,
            observation=observation,
            out_names=out_names,
            selected_indices=selected_indices,
            use_emulator=use_emulator,
            out_dir=out_dir,
        )
        # Class specific parameters
        self.prior_samples = prior_samples
        self.log_likes = None

        # Result parameters
        self.likelihoods = None
        self.log_bme = None
        self.kld = None
        self.inf_entropy = None
        self.post_exp_prior = None
        self.post_exp_likelihoods = None
        self.log_prior_likelihoods = None
        self.posterior = None
        self.ess = None

    def run_sampler(
        self,
        outputs=None,
        std_outputs=None,
        surr_error=None,
        consider_samplesize=False,
        recalculate_loglik=False,
    ) -> pd.DataFrame:
        """
        Performs rejection sampling to update the prior distribution on the
        input parameters.
        If the likelihood is not given to the object, it will be calculated using the
        additional inputs in this function.

        Parameters
        ----------
        outputs : dict, optional
            The metamodel outputs as an array of shape
            (n_samples, n_measurement) for each model output.
            The default is None.
        std_outputs : dict of 2d np arrays, optional
            Standard deviation (uncertainty) associated to the output.
            The default is None.
        surr_error : dict, optional
            A dictionary containing the root mean squared error as array of
            shape (n_samples, n_measurement) for each model output. The default
            is None.
        consider_samplesize : bool, optional
            If set to True will stop the sampler if the effective sample size is
            to small and return None. The default is False
        recalculate_loglik : bool, optional
            If set to True will recalculate the log_likelihood, even if it is
            already given. The default is False.

        Raises
        ------
        AttributeError

        Returns
        -------
        posterior : pd.DataFrame
            Posterior samples of the input parameters.

        """
        # Check for prior samples
        if self.prior_samples is None:
            raise AttributeError("No prior samples available!")

        # Check for likelihoods
        if self.log_likes is None or recalculate_loglik is True:
            if outputs is not None:
                self.log_likes, self.log_bme = self.calculate_loglik_logbme(
                    outputs, rmse=surr_error, std_outputs=std_outputs
                )
            else:
                raise AttributeError("No log-likelihoods available!")

        # Reformat likelihoods
        if len(self.log_likes.shape) == 1:
            self.log_likes = np.swapaxes(np.array([self.log_likes]), 0, 1)
        log_likes = self.log_likes[:, 0]
        likelihoods = np.exp(log_likes, dtype=self.dtype)

        # Check the Effective Sample Size (1<ESS<MCsize)
        self.ess = 1 / np.sum(np.square(likelihoods / np.sum(likelihoods)))

        # Stop if samples don't fulfill the criteria
        n_samples = len(likelihoods)
        if (self.ess > n_samples) or (self.ess < 1):
            warnings.warn(
                f"The effective sample size is low ({self.ess}), \
                          provide more samples to the RejectionSampler."
            )
            if consider_samplesize:
                return None

        # Normalize based on min if all Likelihoods are zero
        norm_likelihoods = likelihoods / np.max(likelihoods)
        if all(likelihoods == 0.0):
            norm_likelihoods = likelihoods / np.min(likelihoods)

        # Reject the poorly performed prior compared to a uniform distribution
        unif = np.random.rand(1, n_samples)[0]
        accepted = norm_likelihoods >= unif
        accepted_samples = self.prior_samples[accepted]

        # Save the accepted likelihood and calculate posterior statistics
        self.likelihoods = likelihoods
        self.post_exp_likelihoods = np.mean(np.log(likelihoods[accepted]))
        self.log_bme = np.log(np.nanmean(self.likelihoods))
        self.posterior = pd.DataFrame(accepted_samples, columns=None)

        return self.posterior

    def calculate_valid_metrics(
        self, exp_design=None, parallel=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate validation metrics including logBME, infEntropy, KLD.

        Parameters
        ----------
        exp_design : obj, optional
            Object of class bvr.ExpDesigns.
            The default is None.
        parallel : bool, optional
            Use multiprocessing in calculation if set to True.
            The default is False.

        Returns
        -------
        kld : np.ndarray
            KLD
        inf_entropy : np.ndarray
            Information entropy

        """
        # Check if likelihoods exist
        if self.log_bme is None:
            raise AttributeError(
                "Rejection sampling has to be run before calculating the validation metrics."
            )
        if self.log_prior_likelihoods is None and exp_design is None:
            raise AttributeError('The attribute "log_prior_likelihoods" has to be set!')

        # Posterior-based expectation of prior densities
        if exp_design is not None:
            try:
                if not parallel:
                    self.post_exp_prior = np.mean(
                        np.log(exp_design.j_dist.pdf(self.posterior.T))
                    )
                else:
                    n_thread = int(0.875 * multiprocessing.cpu_count())
                    with multiprocessing.Pool(n_thread) as p:
                        self.post_exp_prior = np.mean(
                            np.concatenate(
                                p.map(
                                    exp_design.j_dist.pdf,
                                    np.array_split(self.posterior.T, n_thread, axis=1),
                                )
                            )
                        )
            except ValueError:
                self.post_exp_prior = np.mean(self.log_prior_likelihoods)
        else:
            self.post_exp_prior = np.mean(self.log_prior_likelihoods)

        # Calculate Kullback-Leibler Divergence
        # KLD = np.mean(np.log(likelihoods[likelihoods!=0])- logBME)
        kld = self.post_exp_likelihoods - self.log_bme

        # Information Entropy based on Entropy paper Eq. 38
        inf_entropy = self.log_bme - self.post_exp_prior - self.post_exp_likelihoods

        return kld, inf_entropy
