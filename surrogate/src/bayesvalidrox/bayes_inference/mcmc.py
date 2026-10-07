#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostSampler for MCMC sampling.
"""

import os
import multiprocessing
import shutil
import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from emcee.moves import (
    DEMove,
    StretchMove,
    GaussianMove,
    KDEMove,
    DESnookerMove,
    WalkMove,
)

from ..surrogate_models.supplementary import (
    check_ranges,
    gelman_rubin,
)
from .post_sampler import PostSampler

os.environ["OMP_NUM_THREADS"] = "1"


class MCMC(PostSampler):
    """
    A class for bayesian inference via a Markov-Chain Monte-Carlo (MCMC)
    Sampler to approximate the posterior distribution of the Bayes theorem:
    $$p(\\theta|\\mathcal{y}) = \\frac{p(\\mathcal{y}|\\theta) p(\\theta)}
                                         {p(\\mathcal{y})}.$$

    This class make inference with emcee package [1] using an Affine Invariant
    Ensemble sampler (AIES) [2].

    [1] Foreman-Mackey, D., Hogg, D.W., Lang, D. and Goodman, J., 2013.emcee:
        the MCMC hammer. Publications of the Astronomical Society of the
        Pacific, 125(925), p.306. https://emcee.readthedocs.io/en/stable/

    [2] Goodman, J. and Weare, J., 2010. Ensemble samplers with affine
        invariance. Communications in applied mathematics and computational
        science, 5(1), pp.65-80.


    Attributes
    ----------
    engine :  bvr.Engine
        Engine object that contains the surrogate, model and exp_design.
    mcmc_params : dict
        Dictionary of parameters for the mcmc. Required are
        - prior_samples: np.array of size [Nsamples, ndim]
            With samples from the parameters to infer. If given, the walkers will
            be initialized with values sampled equally spaced based on the boundaries
            of the samples given here. No burnin will be done.
            Default is None - in which case the walkers are initialized randomly,
            with a burn in period.
        - n_steps: int
            Number of steps/samples to generate for each walker
        - n_walkers: int
            Number of walkers/independent chains in the ensemble
        - n_burn: int
            Number of samples to consider in the burnin period
        - moves: Obj
            sampling strategy which determines how new points are proposed.
            Must be a valid emcee move object.
            The following options are available
            (see the EMCEE website for more details
            https://emcee.readthedocs.io/en/stable/user/moves/):
                * emcee.moves.KDEMove()
                * emcee.moves.DEMove()
                * emcee.moves.StretchMove()
                * emcee.moves.DESnookerMove()
                * emcee.moves.WalkMove()
                * None - default value. If None is given, then EMCEE uses the
                  StretchMove() by default
        - multiplrocessing: bool
            True to parallelize the different walkers. Default is False
        - verbose: bool
    observation : obj
        An object of class bvr.Observation that contains the observation data
        and related uncertainty.
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
    out_dir : string
        Directory to write the outputs to.

    """

    def __init__(
        self,
        engine,
        mcmc_params,
        observation,
        out_names=None,
        selected_indices=None,
        use_emulator=False,
        out_dir="",
    ):
        # Use parent init
        super().__init__(
            engine,
            observation,
            out_names,
            selected_indices,
            use_emulator,
            out_dir,
        )

        # Param inits
        self.counter = 0

        # Get MCMC parameters from BayesOpts
        self.prior_samples = mcmc_params["prior_samples"]
        if isinstance(self.prior_samples, pd.DataFrame):
            self.prior_samples = self.prior_samples.values
        self.nsteps = int(mcmc_params["n_steps"])
        self.nwalkers = int(mcmc_params["n_walkers"])
        self.nburn = mcmc_params["n_burn"]
        self.moves = mcmc_params["moves"]
        self.mp = mcmc_params["multiprocessing"]
        self.verbose = mcmc_params["verbose"]

        # Check inputs:
        self.is_valid_move()

    def run_sampler(self) -> pd.DataFrame:
        """
        Run the MCMC sampler for the given observations and stdevs.

        Returns
        -------
        Posterior_df : pd.DataFrame
            Posterior samples of the input parameters.

        """
        # Get init values
        engine = self.engine
        n_cpus = engine.model.n_cpus
        ndim = engine.exp_design.ndim
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Set initial samples
        np.random.seed(0)
        if self.prior_samples is None:
            try:
                # Version 1
                prior_dist = self.engine.exp_design.j_dist
                initsamples = prior_dist.sample(self.nwalkers).T
            except:
                # when aPCE selected - gaussian kernel distribution
                input_samples = engine.exp_design.raw_data.T
                random_indices = np.random.choice(
                    len(input_samples), size=self.nwalkers, replace=False
                )
                initsamples = input_samples[random_indices]

            # Check if ndim == 1, change to 2D vector (nwalkers, ndim)
            if initsamples.ndim == 1:
                initsamples = initsamples.reshape(-1, 1)

        else:
            if self.prior_samples.ndim == 1:
                # When MAL is given: perturb the given sample and initialize each walker.
                theta = self.prior_samples
                initsamples = [
                    theta + 1e-1 * np.multiply(np.random.randn(ndim), theta)
                    for i in range(self.nwalkers)
                ]
            else:
                # Pick samples based on a uniform dist between min and max of each dim
                initsamples = np.zeros((self.nwalkers, ndim))
                bound_tuples = []
                for idx_dim in range(ndim):
                    lower = np.min(self.prior_samples[:, idx_dim])
                    upper = np.max(self.prior_samples[:, idx_dim])
                    bound_tuples.append((lower, upper))
                    dist = st.uniform(loc=lower, scale=upper - lower)
                    initsamples[:, idx_dim] = dist.rvs(size=self.nwalkers)

                # Update lower and upper
                engine.exp_design.bound_tuples = bound_tuples

        print("\n>>>> Bayesian inference with MCMC started. <<<<<<")

        # Set up the backend and clear it in case the file already exists
        backend = emcee.backends.HDFBackend(f"{self.out_dir}/emcee_sampler.h5")
        backend.reset(self.nwalkers, ndim)

        # Define emcee sampler
        # Here we'll set up the computation. emcee combines multiple "walkers",
        # each of which is its own MCMC chain. The number of trace results will
        # be nwalkers * nsteps.
        if self.mp:
            # Run in parallel
            if n_cpus is None:
                n_cpus = multiprocessing.cpu_count()

            with multiprocessing.Pool(n_cpus) as pool:
                sampler = emcee.EnsembleSampler(
                    self.nwalkers,
                    ndim,
                    self.log_posterior,
                    moves=self.moves,
                    pool=pool,
                    backend=backend,
                )

                # Check if a burn-in phase is needed!
                if self.prior_samples is None:
                    # Burn-in
                    print("\n Burn-in period is starting:")
                    pos = sampler.run_mcmc(initsamples, self.nburn, progress=True)

                    # Reset sampler
                    pos = pos.coords
                    sampler.reset()
                else:
                    pos = initsamples

                # Production run
                print("\n Production run is starting:")
                pos, _, _ = sampler.run_mcmc(pos, self.nsteps, progress=True)

        else:
            # Run in series and monitor the convergence
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                ndim,
                self.log_posterior,
                moves=self.moves,
                backend=backend,
                vectorize=True,
            )
            print(f"ndim: {ndim}")
            print(f"initsamples.shape: {initsamples.shape}")
            # Check if a burn-in phase is needed!
            if self.prior_samples is None:
                # Burn-in
                print("\n Burn-in period is starting:")
                pos = sampler.run_mcmc(initsamples, self.nburn, progress=True)

                # Reset sampler
                sampler.reset()
                pos = pos.coords
            else:
                pos = initsamples

            # Production run
            print("\n Production run is starting:")

            # Track how the average autocorrelation time estimate changes
            autocorr_idx = 0
            autocorr = np.empty(self.nsteps)
            tauold = np.inf
            autocorreverynsteps = 50

            # Sample step by step using the generator sampler.sample
            for _ in sampler.sample(
                pos, iterations=self.nsteps, tune=True, progress=True
            ):

                # Only check convergence every autocorreverynsteps steps
                if sampler.iteration % autocorreverynsteps:
                    continue

                # Print the current mean acceptance fraction
                if self.verbose:
                    print(f"\nStep: {sampler.iteration}")
                    acc_fr = np.mean(sampler.acceptance_fraction)
                    print(f"Mean acceptance fraction: {acc_fr:.3f}")

                # Compute the autocorrelation time so far
                # using tol=0 means that we'll always get an estimate even if
                # it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                # Average over walkers
                autocorr[autocorr_idx] = np.nanmean(tau)
                autocorr_idx += 1

                # Output current autocorrelation estimate
                if self.verbose:
                    print(f"Mean autocorr. time estimate: {np.nanmean(tau):.3f}")
                    list_gr = np.round(gelman_rubin(sampler.chain), 3)
                    print("Gelman-Rubin Test*: ", list_gr)

                # Check convergence
                converged = np.all(tau * autocorreverynsteps < sampler.iteration)
                converged &= np.all(np.abs(tauold - tau) / tau < 0.01)
                converged &= np.all(gelman_rubin(sampler.chain) < 1.1)

                if converged:
                    break
                tauold = tau

        # Posterior diagnostics
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except emcee.autocorr.AutocorrError:
            tau = 5

        if all(np.isnan(tau)):
            tau = 5

        burnin = int(2 * np.nanmax(tau))
        thin = int(0.5 * np.nanmin(tau)) if int(0.5 * np.nanmin(tau)) != 0 else 1
        finalsamples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        acc_fr = np.nanmean(sampler.acceptance_fraction)
        list_gr = np.round(gelman_rubin(sampler.chain[:, burnin:]), 3)

        # Print summary
        print("\n")
        print("-" * 15 + "Posterior diagnostics" + "-" * 15)
        print(f"Mean auto-correlation time: {np.nanmean(tau):.3f}")
        print(f"Thin: {thin}")
        print(f"Burn-in: {burnin}")
        print(f"Flat chain shape: {finalsamples.shape}")
        print(f"Mean acceptance fraction*: {acc_fr:.3f}")
        print(f"Gelman-Rubin Test**: {list_gr}")

        print("\n* This value must lay between 0.234 and 0.5.")
        print("** These values must be smaller than 1.1.")
        print("-" * 50)

        print("\n>>>> Bayesian inference with MCMC successfully completed. <<<<<<\n")

        # Extract parameter names and their prior ranges
        par_names = engine.exp_design.par_names
        params_range = engine.exp_design.bound_tuples

        # Plot traces
        if self.verbose and self.nsteps < 10000:
            for par_idx in range(ndim):
                # Set up the axes with gridspec
                fig = plt.figure()
                grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
                main_ax = fig.add_subplot(grid[:-1, :3])
                y_hist = fig.add_subplot(grid[:-1, -1], xticklabels=[], sharey=main_ax)

                for i in range(self.nwalkers):
                    samples = sampler.chain[i, :, par_idx]
                    main_ax.plot(samples, "-")

                    # histogram on the attached axes
                    y_hist.hist(
                        samples[burnin:],
                        40,
                        histtype="stepfilled",
                        orientation="horizontal",
                        color="gray",
                    )

                main_ax.set_ylim(params_range[par_idx])
                main_ax.set_title("traceplot for " + par_names[par_idx])
                main_ax.set_xlabel("step number")

                # save the current figure
                fig.savefig(self.out_dir + "/traceplots.pdf", bbox_inches="tight")

                # Destroy the current plot
                plt.close()

        # plot development of autocorrelation estimate
        if not self.mp:
            fig1 = plt.figure()
            steps = autocorreverynsteps * np.arange(1, autocorr_idx + 1)
            taus = autocorr[:autocorr_idx]
            plt.plot(steps, steps / autocorreverynsteps, "--k")
            plt.plot(steps, taus)
            plt.xlim(0, steps.max())
            plt.ylim(0, np.nanmax(taus) + 0.1 * (np.nanmax(taus) - np.nanmin(taus)))
            plt.xlabel("number of steps")
            plt.ylabel(r"mean $\hat{\tau}$")
            fig1.savefig(
                f"{self.out_dir}/autocorrelation_time.pdf", bbox_inches="tight"
            )

        posterior_df = pd.DataFrame(finalsamples, columns=par_names)

        return posterior_df

    # -------------------------------------------------------------------------
    def log_prior(self, theta) -> np.ndarray:
        """
        Calculates the log prior likelihood \\( p(\\theta)\\) for the given
        parameter set(s) \\( \\theta \\).

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter sets, i.e. proposals of MCMC chains.

        Returns
        -------
        logprior: np.ndarray
            Log prior likelihood. If theta has only one row, a single value is
            returned otherwise an array of shape (n_samples) is returned.

        """
        engine = self.engine

        # Find the number of sigma2 parameters
        n_sigma2 = -len(theta)
        prior_dist = engine.exp_design.prior_space
        params_range = engine.exp_design.bound_tuples
        theta = theta if theta.ndim != 1 else theta.reshape((1, -1))
        nsamples = theta.shape[0]
        logprior = -np.inf * np.ones(nsamples)

        for i in range(nsamples):
            # Check if the sample is within the parameters' range
            if check_ranges(theta[i], params_range):
                # Check if all dists are uniform, if yes priors are equal.
                if all(
                    engine.exp_design.input_object.marginals[i].dist_type == "uniform"
                    for i in range(engine.exp_design.ndim)
                ):
                    logprior[i] = 0.0
                else:
                    logprior[i] = np.ravel(np.log(prior_dist.pdf(theta[i, :-n_sigma2].T)))[0]

        if nsamples == 1:
            return logprior[0]
        return logprior

    # -------------------------------------------------------------------------
    def log_likelihood(self, theta) -> np.ndarray:
        """
        Computes likelihood \\( p(\\mathcal{Y}|\\theta)\\) of the performance
        of the (meta-)model in reproducing the observation data.

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter set, i.e. proposals of the MCMC chains.

        Returns
        -------
        log_like : np.ndarray
            Log likelihood. Shape: (n_samples)

        """
        # Find the number of sigma2 parameters
        theta = theta if theta.ndim != 1 else theta.reshape((1, -1))

        # Evaluate Model/MetaModel at theta
        mean_pred, _ = self.eval_model(theta)

        # Surrogate model's error using RMSE of test data
        surr_error = None
        if self.use_emulator:
            if "rmse" not in self.engine.valid_metrics.keys():
                raise AttributeError("The given engine has no rmse value.")
            surr_error = {}
            for key in self.engine.out_names:
                surr_error[key] = self.engine.valid_metrics["rmse"][key][-1]

        # Likelihood
        log_like = self.normpdf(
            mean_pred,
            rmse=surr_error,
        )
        return log_like

    # -------------------------------------------------------------------------
    def log_posterior(self, theta) -> np.ndarray:
        """
        Computes the posterior likelihood \\(p(\\theta| \\mathcal{Y})\\) for
        the given parameterset.

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter set, i.e. proposals of the MCMC chains.

        Returns
        -------
        log_like : np.ndarray
            Log posterior likelihood. Shape: (n_samples)

        """

        nsamples = 1 if theta.ndim == 1 else theta.shape[0]

        if nsamples == 1:
            if self.log_prior(theta) == -np.inf:
                return -np.inf

            # Compute log Likelihood
            log_prior = self.log_prior(theta)
            log_likelihood = self.log_likelihood(theta)
            return log_prior + log_likelihood

        # Compute log prior
        log_prior = self.log_prior(theta)

        # Initialize log_likelihood
        log_likelihood = -np.inf * np.ones(nsamples)

        # Find the indices for -inf sets
        non_inf_idx = np.where(log_prior != -np.inf)[0]

        # Compute loLikelihoods
        if non_inf_idx.size != 0:
            log_likelihood[non_inf_idx] = self.log_likelihood(theta[non_inf_idx])

        return log_prior + log_likelihood

    # -------------------------------------------------------------------------
    def eval_model(self, theta) -> tuple[dict, dict]:
        """
        Evaluates the (meta-) model at the given theta.

        Parameters
        ----------
        theta : array of shape (n_samples, n_params)
            Parameter set, i.e. proposals of the MCMC chains.

        Returns
        -------
        mean_pred : dict
            Mean model prediction.
        std_pred : dict
            Std of model prediction.

        """
        engine = self.engine
        model = engine.model

        if self.use_emulator:
            # Evaluate the MetaModel
            mean_pred, std_pred = engine.meta_model.eval_metamodel(theta)
        else:
            # Evaluate the origModel
            mean_pred, std_pred = {}, {}

            model_outs, _ = model.run_model_parallel(
                theta,
                prev_run_no=self.counter,
                key_str="_MCMC",
                mp=False,
                verbose=False,
            )

            # Save outputs in respective dicts
            for _, var in enumerate(self.out_names):
                mean_pred[var] = model_outs[var]
                std_pred[var] = np.zeros((mean_pred[var].shape))

            # Remove the folder
            if model.link_type.lower() != "function":
                shutil.rmtree(f"{model.name}_MCMC_{self.counter+1}")

            # Add one to the counter
            self.counter += 1

        return mean_pred, std_pred

    def is_valid_move(self) -> bool:
        """
        Checks to see if user-provided Move is a valid EMCEE move.

        Raises
        ------
        ValueError

        Returns
        -------
        bool
            True if valid move, raises an error if not a valid move.
        """
        valid_moves = (
            DEMove,
            StretchMove,
            GaussianMove,
            KDEMove,
            DESnookerMove,
            WalkMove,
        )
        if isinstance(self.moves, valid_moves) or self.moves is None:
            return True
        raise ValueError(
            "An invalid Move was provided for the MCMC iterations.\
                Please use a valid EMCEE move."
        )
