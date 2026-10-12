# -*- coding: utf-8 -*-
"""
Sequential design to choose new training samples
"""
import sys
from functools import wraps
from copy import deepcopy
import multiprocessing
import warnings
from joblib import Parallel, delayed
import numpy as np
import scipy.optimize as opt
from scipy import stats, signal, linalg, sparse
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ..bayes_inference.rejection_sampler import RejectionSampler
from .exploration import Exploration
from .supplementary import create_psi, subdomain
from .gaussian_process_sklearn import GPESkl


def _loop_exploit(exploit_func):
    """
    Run exploit-function directly if using dual annealing,
    else loop through the samples.

    Note:
    Function args should only include the candidate samples.
    All other function inputs should be set as kwargs.

    Parameters
    ----------
    fit_function : function
        Function with the same signature as self.fit().

    Returns
    -------
    decorator : function
        The decorated function.

    """

    @wraps(exploit_func)
    def decorator(self, *args, **kwargs):
        # No loop with dual annealing
        if self.exp_design.explore_method == "dual_annealing":
            return exploit_func(self, args[0], **kwargs)

        # Loop through candidates
        candidates = args[0]
        try:
            index = args[1]
        except IndexError:
            index = 0
        scores = np.zeros(candidates.shape[0])
        for idx, x_can in tqdm(enumerate(candidates), ascii=True, desc="Exploitation"):
            scores[idx] = np.ravel(exploit_func(self, np.array([x_can]), **kwargs))[0]

        return -1 * scores, index

    return decorator


class SequentialDesign:
    """
    Contains options for choosing the next training sample iteratively.

    Parameters
    ----------
    meta_model : obj
        A bvr.MetaModel object. If no MetaModel should be trained and used, set
        this to None.
    model : obj
        A model interface of type bvr.PyLinkForwardModel that can be run.
    exp_design : obj
        The experimental design that will be used to sample from the input
        space.
    observation : obj
        An object of class bvr.Observation that contains the observation data
        and related uncertainty.
    parallel : bool, optional
        Set to True if the evaluations should be done in parallel.
        The default is False.
    out_names : list, optional
        The list of requested output keys to be used for the analysis.
        The default is `None`.
    verbose : bool, optional
        Verbosity of the methods, the default is False.
    """

    def __init__(
        self,
        meta_model,
        exp_design,
        observation,
        parallel=False,
        out_names=None,
        verbose=False,
    ):
        self.meta_model = meta_model
        self.exp_design = exp_design
        self.observation = observation
        self.parallel = parallel
        self.out_names = out_names if out_names is not None else []
        self.verbose = verbose

        # Init other parameters
        self.bound_tuples = []
        self.mc_samples = None
        self.results = None
        self.likelihoods = None
        self.prev_mm = None
        self.explore = None
        self.x_mc = None
        self.mc_size = 15000
        self.rej_sampler = None

    # -------------------------------------------------------------------------

    def choose_next_sample(self):
        """
        Runs optimal sequential design.

        Raises
        ------
        NameError
            Wrong utility function.

        Returns
        -------
        Xnew : array (n_samples, n_params)
            Selected new training point(s).

        """
        # Initialization
        bounds = self.exp_design.bound_tuples
        n_new_samples = self.exp_design.n_new_samples
        explore_method = self.exp_design.explore_method
        exploit_method = self.exp_design.exploit_method
        tradeoff_scheme = self.exp_design.tradeoff_scheme
        n_candidates = self.exp_design.n_candidates

        old_ed_x = self.exp_design.x
        old_ed_y = self.exp_design.y.copy()
        ndim = self.exp_design.x.shape[1]

        # ----------- CUSTOMIZED METHODS ----------
        # Here exploration and exploitation are performed simulataneously
        if explore_method.lower() == "dual_annealing":

            # Divide the domain to subdomains
            subdomains = subdomain(bounds, n_new_samples)

            # Run the dual annealing
            results = []
            if self.parallel:
                args = []
                for i in range(n_new_samples):
                    args.append((exploit_method, subdomains[i], i))
                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:

                    # With Pool.starmap_async()
                    results = pool.starmap_async(self.dual_annealing, args).get()

                # Close the pool
                # pool.close()
            else:
                for i in range(n_new_samples):
                    results.append(
                        self.dual_annealing(exploit_method, subdomains[i], i)
                    )

            # New sample
            x_new = np.array([results[i][1] for i in range(n_new_samples)])
            if self.verbose:
                print("")
                print("\nXnew:\n", x_new)
            return x_new

        # ------- Tradeoff weights -------
        # Compute exploration weight based on trade off scheme
        explore_w, exploit_w = self.tradeoff_weights(
            tradeoff_scheme, old_ed_x, old_ed_y
        )
        print(
            f"\n Exploration weight={explore_w:0.3f} "
            f"Exploitation weight={exploit_w:0.3f}\n"
        )

        # ---------- EXPLORATION METHODS ----------
        norm_score_exploration = None
        if explore_method.lower() == "loocv":
            # -----------------------------------------------------------------
            # 'LOOCV': only works with PCE
            # Creates error model using loocv error during surrogate training.
            # Evaluates error model on exploration candidates
            # Sets exploration score based on estimated loocv error

            # Generate random samples
            all_candidates = self.exp_design.generate_samples(n_candidates, "random")

            if "pce" in self.meta_model.meta_model_type.lower():
                if not hasattr(self.meta_model, "lc_error"):
                    raise AttributeError(
                        f"The meta_model {self.meta_model.meta_model_type} does not have "
                        "a 'lc_error' attribute. Please check the meta_model type."
                    )
                all_values = []

                # Average over all levels of the lc_error dictionary
                lc_error_dict = self.meta_model.lc_error
                for b_i in lc_error_dict.values():  # level1
                    for subdict in b_i.values():  # level2
                        level3_arrays = list(subdict.values())
                        level3_stack = np.stack(level3_arrays, axis=1)
                        level2_avg = np.mean(level3_stack, axis=1)
                        all_values.append(level2_avg)

                # Stack all level2 averages and compute mean over axis=0
                all_values_stack = np.stack(all_values, axis=1)
                lc_error = {"e": np.mean(all_values_stack, axis=1).reshape(-1, 1)}

                # Train error model
                error_model = GPESkl(
                    input_obj=self.meta_model.input_obj,
                    kernel_type="RBF",
                    isotropy=True,  # keep constant
                    noisy=True,  # keep constant
                    verbose=False,
                    n_bootstrap_itrs=1,
                    dim_red_method="no",
                )
                error_model.fit(old_ed_x, lc_error, parallel=False, verbose=False)

                # Evaluate error model on all candidates
                e_lc_all_cands, _ = error_model.eval_metamodel(all_candidates)
                e_lc_all_cands = e_lc_all_cands["e"].reshape(
                    -1,
                )

                # Shift scores so all values are positive
                shifted_scores = e_lc_all_cands - np.nanmin(e_lc_all_cands)

                # Normalize the error w.r.t the sum
                norm_score_exploration = shifted_scores / np.nansum(shifted_scores)

            else:
                raise AttributeError(
                    "The meta_model does not have a 'lc_error' attribute. "
                    "Please check the meta_model type."
                )
        else:
            if explore_method.lower() not in [
                "random",
                "voronoi",
                "sobol",
                "latin_hypercube",
            ]:
                warnings.warn(
                    f"The exploration method '{explore_method}' is not officially "
                    "supported. Switching to random exploration."
                )
                self.exp_design.explore_method = "random"

            # ------- EXPLORATION: SPACE-FILLING DESIGN -------
            # Generate candidate samples from Exploration class
            self.explore = Exploration(
                self.exp_design, n_candidates, verbose=self.verbose
            )
            # Select criterion (mc-intersite-proj-th, mc-intersite-proj)
            self.explore.mc_criterion = "mc-intersite-proj"
            (
                all_candidates,
                norm_score_exploration,
            ) = self.explore.get_exploration_samples()

        # --------- EXPLOITATION METHODS ----------
        norm_score_exploitation = None
        if tradeoff_scheme.lower() == "explore_only" or explore_w == 1.0:
            norm_score_exploitation = norm_score_exploration
        else:
            norm_score_exploitation = self.run_exploitation(all_candidates)

        # Accumulate all candidates and scores
        final_candidates = all_candidates
        total_score = (
            exploit_w * norm_score_exploitation + explore_w * norm_score_exploration
        )

        # ------- Select the best candidate -------
        # Find the optimal point subset to add to the initial design by
        # maximization of the utility score and taking care of NaN values
        temp = total_score.copy()

        # Since we are maximizing
        temp[np.isnan(total_score)] = -np.inf
        sorted_idxtotal_score = np.argsort(temp)[::-1]
        best_idx = sorted_idxtotal_score[:n_new_samples]
        if isinstance(best_idx, int):
            best_idx = [best_idx]

        # Select the requested number of samples
        if explore_method.lower() == "voronoi":
            x_new = np.zeros((n_new_samples, ndim))
            for i, idx in enumerate(best_idx):
                x_can = self.explore.closest_points[idx]

                # Calculate the maxmin score for the region of interest
                new_samples, maxmin_score = self.explore.get_mc_samples(x_can)

                # Select the requested number of samples
                x_new[i] = new_samples[np.argmax(maxmin_score)]
        else:
            x_new = final_candidates[sorted_idxtotal_score[:n_new_samples]]

        if self.verbose:
            print("")
            print(f"Run No. {old_ed_x.shape[0] + 1} finished, ")
            print("New training point:\n", x_new)

        return x_new

    # -------------------------------------------------------------------------
    def tradeoff_weights(self, tradeoff_scheme, old_ed_x, old_ed_y):
        """
        Calculates weights for exploration scores based on the requested
        scheme: `None`, `equal`, `epsilon-decreasing` and `adaptive`.

        `exploit_only`: No exploration, full exploitation.
        `explore_only`: Full exploration, no exploitation.
        `equal`: Same weights for exploration and exploitation scores.
        `epsilon-decreasing`: Start with more exploration and increase the
            influence of exploitation along the way with an exponential decay
            function
        `adaptive`: An adaptive method based on:
            Liu, Haitao, Jianfei Cai, and Yew-Soon Ong. "An adaptive sampling
            approach for Kriging metamodeling by maximizing expected prediction
            error." Computers & Chemical Engineering 106 (2017): 171-182.

        Parameters
        ----------
        tradeoff_scheme : string
            Trade-off scheme for exloration and exploitation scores.
        old_ed_x : array (n_samples, n_params)
            Old experimental design (training points).
        old_ed_y : dict
            Old model responses (targets).

        Returns
        -------
        exploration_weight : float
            Exploration weight.
        exploitation_weight: float
            Exploitation weight.

        """
        exploration_weight = None
        init_n_samples = self.exp_design.n_init_samples
        n_max_samples = self.exp_design.n_max_samples

        if tradeoff_scheme.lower() == "exploit_only":
            exploration_weight = 0

        elif tradeoff_scheme.lower() == "explore_only":
            exploration_weight = 1

        elif tradeoff_scheme.lower() == "equal":
            exploration_weight = 0.5

        elif tradeoff_scheme.lower() == "epsilon-decreasing":
            itr_number = self.exp_design.x.shape[0] - init_n_samples
            itr_number //= self.exp_design.n_new_samples
            n_it_tot = n_max_samples - init_n_samples
            if n_it_tot == 0:
                exploration_weight = 1
            else:
                tau2 = -(n_max_samples - init_n_samples - 1) / np.log(1e-8)
                exploration_weight = signal.windows.exponential(
                    n_max_samples - init_n_samples, 0, tau2, False
                )[itr_number]

        elif tradeoff_scheme.lower() == "adaptive":
            itr_number = self.exp_design.x.shape[0] - init_n_samples
            itr_number //= self.exp_design.n_new_samples

            if itr_number == 0:
                exploration_weight = 0.5
            else:
                # Get last design point
                last_ed_x = old_ed_x[-1].reshape(1, -1)
                y = np.array(list(old_ed_y.values()))[:, -1, :]

                # Evaluate in previous mm
                y_hat_prev, _ = self.prev_mm.eval_metamodel(last_ed_x)
                metamod_y_prev = np.array(list(y_hat_prev.values()))[:, 0]
                mse_cv_error = mean_squared_error(metamod_y_prev, y)

                # Evaluate in current mm
                last_out_y, _ = self.meta_model.eval_metamodel(last_ed_x)
                pce_y = np.array(list(last_out_y.values()))[:, 0]
                mse_error = mean_squared_error(pce_y, y)

                # Calculate the exploration weight
                exploration_weight = min([0.5 * mse_error / mse_cv_error, 1])

        else:
            raise AttributeError(
                f"The chosen tradeoff scheme {tradeoff_scheme} is not supported."
            )

        # Calculate exploitation weight
        exploitation_weight = 1 - exploration_weight

        return exploration_weight, exploitation_weight

    # -------------------------------------------------------------------------
    def run_exploitation(self, all_candidates):
        """
        Run the selected exploitation method.
        Supports 'bayesactdesign', 'varoptdesign' and 'alphabetic'.

        Parameters
        ----------
        all_candidates : np.ndarray
            Candidate samples.

        Returns
        -------
        norm_score_exploitation : np.ndarray
            Exploitation scores
        opt_type : str
            Optimization type, either 'minimization' or 'maximization'

        """
        explore_method = self.exp_design.explore_method
        exploit_method = self.exp_design.exploit_method
        n_cand_groups = self.exp_design.n_cand_groups
        n_candidates = self.exp_design.n_candidates

        norm_score_exploitation = None
        if exploit_method.lower() in [
            "bayesactdesign",
            "varoptdesign",
        ]:

            # Select util function
            util_f = self.util_bayesian_active_design
            if exploit_method.lower() == "varoptdesign":
                util_f = self.util_var_opt_design
                self.x_mc = self.exp_design.generate_samples(self.mc_size, "random")
            elif exploit_method.lower() == "bayesactdesign":
                # Check for observations
                if self.observation is None:
                    raise AttributeError(
                        "Bayesian active design can only be run if an observation is given!"
                    )
                self.rej_sampler = RejectionSampler(
                    out_names=self.out_names,
                    use_emulator=False,
                    observation=self.observation,
                )

            # Parallelize the exploitation method or serially run it
            if self.parallel:
                # Split the candidates in groups for multiprocessing
                split_cand = np.array_split(all_candidates, n_cand_groups, axis=0)
                if explore_method.lower() != "voronoi":
                    split_cand = np.array_split(all_candidates, n_cand_groups, axis=0)
                    good_sample_idx = range(n_cand_groups)
                else:
                    # Find indices of the Vornoi cells with samples
                    good_sample_idx = []
                    for idx, _ in enumerate(self.explore.closest_points):
                        if len(self.explore.closest_points[idx]) != 0:
                            good_sample_idx.append(idx)
                    split_cand = self.explore.closest_points
                    n_cand_groups = len(good_sample_idx)

                # Run parallelized exploitation method
                results = Parallel(n_jobs=-1, backend="multiprocessing")(
                    delayed(util_f)(split_cand[i], i) for i in range(n_cand_groups)
                )

                # Retrieve the results and append them
                scores = np.concatenate([results[k][0] for k in range(n_cand_groups)])

            else:
                scores, _ = util_f(all_candidates, 0)

            # Check if all scores are inf
            if np.isinf(scores).all() or np.isnan(scores).all():
                scores = np.ones(len(scores))

            # Get the expected value (mean) of the Utility score per cell
            if explore_method.lower() == "voronoi":
                scores = np.mean(scores.reshape(-1, n_candidates), axis=1)

            # Normalize scores
            # Shift scores to make them all positive
            scores_shifted = scores - np.nanmin(scores)
            # Normalize with shifted scores
            norm_score_exploitation = scores_shifted / np.nansum(scores_shifted)
            # norm_score_exploitation = scores / np.abs(
            #     np.nansum(scores)
            # )  # abs makes sure the sign is kept

        elif exploit_method.lower() == "alphabetic":
            norm_score_exploitation = self.util_alph_opt_design(all_candidates)

        else:
            raise NameError(
                f"The requested exploitation method {exploit_method} is not available."
            )

        return norm_score_exploitation

    # -------------------------------------------------------------------------

    @_loop_exploit
    def util_var_opt_design(self, x_can):
        """
        Computes the exploitation scores based on:

        `ALM (Active Learning MacKay)`:
            Selects points where the surrogate model's predictive variance is largest.
        `EIGF (Expected Improvement for Global Fit)`:
            Selects points where the prediction error (vs. nearest observed data)
            plus the variance is largest.
        `MI (Mutual Information)`:
            Selects points that maximize mutual information between the new sample
            and the unobserved.
        `ALC (Active Learning Cohn)`:
            Selects candidate points that minimize the average predictive variance
            across a set of Monte Carlo evaluation points.

        The calculations are based on the following sources.
        ALM:
            MacKay, D. J. (1992). Information-based objective functions
            for active data selection. Neural computation
        EIGF, MI, ALC:
            Beck, J., & Guillas, S. (2016). Sequential design with mutual
            information for computer experiments (MICE): Emulation of a
            tsunami model. SIAM/ASA Journal on Uncertainty Quantification

        Parameters
        ----------
        x_can : array of shape (n_samples, n_params)
            Candidate samples.

        Returns
        -------
        float
            Score.

        """
        meta_model = self.meta_model
        old_ed_x = self.exp_design.x
        old_ed_y = self.exp_design.y
        out_names = self.out_names
        util_func = self.exp_design.util_func

        # Run the meta_model for the candidate
        x_can = x_can.reshape(1, -1)
        y_metamod_can, std_metamod_can = meta_model.eval_metamodel(x_can)

        score = None
        if util_func.lower() == "alm":  # maximize
            # Compute perdiction variance of the old model
            can_pred_var = {key: std_metamod_can[key] ** 2 for key in out_names}

            var_metamod = np.zeros((len(out_names), x_can.shape[0]))
            for idx, key in enumerate(out_names):
                var_metamod[idx] = np.max(can_pred_var[key], axis=1)
            score = np.max(var_metamod, axis=0)

        elif util_func.lower() == "eigf":  # maximize
            # Find closest EDX to the candidate
            distances = distance.cdist(old_ed_x, x_can, "euclidean")
            index = np.argmin(distances)

            # Compute perdiction error and variance of the old model
            pred_error = {key: y_metamod_can[key] for key in out_names}
            can_pred_var = {key: std_metamod_can[key] ** 2 for key in out_names}

            # Compute perdiction error and variance of the old model
            eigf_metamod = np.zeros((len(out_names), x_can.shape[0]))
            for idx, key in enumerate(out_names):
                residual = pred_error[key] - old_ed_y[key][int(index)]
                var = can_pred_var[key]
                eigf_metamod[idx] = np.max(residual**2 + var, axis=1)
            score = np.max(eigf_metamod, axis=0)

        elif util_func.lower() in ["mi", "alc"]:
            # Evaluate metamodel at the candidate location
            if x_can.ndim == 1:
                x_can = x_can.reshape(1, -1)
            y_metamod_can, y_std_can = meta_model.eval_metamodel(x_can)

            # Update exp_design with the candidates
            new_ed_x = np.vstack((old_ed_x, x_can))
            new_ed_y = {}
            for key in old_ed_y.keys():
                new_ed_y[key] = np.vstack((old_ed_y[key], y_metamod_can[key]))

            # Train the model for the observed data using x_can
            mm_can = deepcopy(meta_model)
            mm_can.fit(new_ed_x, new_ed_y, parallel=self.parallel)

            # Mutual information: maximize
            if util_func.lower() == "mi":

                # Evaluate the meta_model at the given samples
                _, std_metamod_can = mm_can.eval_metamodel(x_can)
                std_can = {key: std_metamod_can[key] for key in out_names}
                std_old = {key: y_std_can[key] for key in out_names}

                # Compute the score
                var_metamod = np.zeros((len(out_names)))
                for i, key in enumerate(out_names):
                    var_metamod[i] = np.mean(std_old[key] ** 2 / std_can[key] ** 2)
                score = np.mean(var_metamod)

            # Active learning Cohn
            if util_func.lower() == "alc":
                # Evaluate the meta_model on mc samples
                _, y_mc_std = meta_model.eval_metamodel(self.x_mc)
                _, y_mc_std_can = mm_can.eval_metamodel(self.x_mc)

                # Compute the score
                score = []
                for i, key in enumerate(out_names):
                    var_old = y_mc_std[key] ** 2  # Before adding candidate
                    var_new = y_mc_std_can[key] ** 2  # After adding candidate
                    score.append(np.mean(var_old - var_new, axis=0))
                score = np.mean(score)  # Maximize score
        else:
            raise AttributeError(
                f"The requested utility function {util_func} is not available for VarOptDesign."
            )

        return -1 * score

    # -------------------------------------------------------------------------
    @_loop_exploit
    def util_bayesian_active_design(self, x_can):
        """
        Computes score based on Bayesian active design criterion (var).

        It is based on the following paper:
        Oladyshkin, Sergey, Farid Mohammadi, Ilja Kroeker, and Wolfgang Nowak.
        "Bayesian3 active learning for the gaussian process emulator using
        information theory." Entropy 22, no. 8 (2020): 890.

        Parameters
        ----------
        x_can : np.ndarray
            A single candidate sample.

        Returns
        -------
        float
            Exploitation score for the candidate sample.

        """
        # Check that observation exists
        if self.rej_sampler.observation is None:
            raise AttributeError(
                "Bayesian active design can only be run if an observation is given!"
            )
        util_func = self.exp_design.util_func

        # Evaluate metamodel on the candidate sample
        x_can = x_can.reshape(1, -1)
        y_mm_can, std_mm_can = self.meta_model.eval_metamodel(x_can)

        # Sample a distribution for a normal dist
        # with Y_mean_can as the mean and Y_std_can as std.
        y_mc, std_mc = {}, {}
        log_prior_likelihoods = np.zeros(self.mc_size)
        prior_samples = None
        for key in list(y_mm_can):
            cov = np.diag(std_mm_can[key][0, :] ** 2)

            # Allow for singular matrices
            rv = stats.multivariate_normal(
                mean=y_mm_can[key][0, :], cov=cov, allow_singular=True
            )

            y_mc[key] = rv.rvs(size=self.mc_size)
            log_prior_likelihoods += rv.logpdf(y_mc[key])
            std_mc[key] = np.zeros((self.mc_size, y_mm_can[key].shape[1]))

            # Save output space samples in an array (n_samples, n_out)
            if prior_samples is None:
                prior_samples = y_mc[key]
            else:
                prior_samples = np.hstack((prior_samples, y_mc[key]))

        #  Likelihood computation (Comparison of data and simulation
        #  results via PCE with candidate design)
        self.rej_sampler.prior_samples = prior_samples  # Send outputs as prior array
        self.rej_sampler.log_prior_likelihoods = log_prior_likelihoods
        x_post = self.rej_sampler.run_sampler(
            outputs=y_mc, std_outputs=std_mc, recalculate_loglik=True
        )
        kld, inf_entropy = self.rej_sampler.calculate_valid_metrics(None)
        likelihoods = self.rej_sampler.likelihoods

        score = None
        # Kullback-Leibler Divergence (Sergey's paper)
        if util_func.lower() == "dkl":
            score = kld  # maximize

        # Marginal log likelihood
        elif util_func.lower() == "bme":
            score = np.nanmean(likelihoods)  # maximize

        # Entropy-based information gain
        elif util_func.lower() == "ie":
            score = inf_entropy * -1  # -1 for maximization

        # Deviance information criterion
        elif util_func.lower() == "dic":

            # Deviance for each posterior draw
            deviance_samples = -2 * np.log(likelihoods[likelihoods > 0])
            # Mean deviance
            mean_deviance = np.mean(deviance_samples)

            # Deviance at posterior mean (approximate using y_mm_can)
            # normpdf(...) should return log-likelihood, not likelihood!
            loglik_theta_mean = self.rej_sampler.normpdf(
                y_mm_can, std_outputs=std_mm_can
            )
            deviance_theta_mean = -2 * loglik_theta_mean
            # Effective number of parameters: variance
            # Before: 0.5 * np.var(np.log(likelihoods[likelihoods != 0]))
            p_d = mean_deviance - deviance_theta_mean

            # DIC definition
            dic = mean_deviance + p_d

            score = -1 * dic  # Minimize (-1* to maximize)

        # Bayes risk likelihood
        elif util_func.lower() == "bayesrisk":
            score = -1 * np.var(likelihoods)  # maximize

        # D-Posterior-precision - covariance of the posterior parameters
        elif util_func.lower() == "dpp":
            if x_post.shape[0] == 1:
                score = np.nan
            else:
                score = -np.log(
                    np.linalg.det(np.cov(x_post, rowvar=False))
                )  # -1* to maximize

        # A-Posterior-precision - trace of the posterior parameters
        elif util_func.lower() == "app":
            if x_post.shape[0] == 1:
                # If only one posterior sample is available, covariance is zero
                score = np.nan
            else:
                score = -np.log(
                    np.trace(np.cov(x_post, rowvar=False))
                )  # -1* to maximize

        else:
            raise AttributeError(
                f"The requested utility function {util_func} is not available for BayesActDesign."
            )

        # Handle inf and NaN (replace by zero)
        if np.isnan(score) or score == -np.inf or score == np.inf:
            score = np.nan  # 0

        # Clear memory
        del likelihoods
        del y_mc
        del std_mc

        return -1 * score

    # -------------------------------------------------------------------------
    def dual_annealing(self, exploit_method, bounds, run_idx, verbose=False):
        """
        Exploration algorithm to find the optimum parameter space.

        Note:
        Currently does not support BayesActDesign due to possibility
        of inf/nan values.

        Parameters
        ----------
        exploit_method : string
            Exploitation method: `VarOptDesign`
        bounds : list of tuples
            List of lower and upper boundaries of parameters.
        run_idx : int
            Run number.
        verbose : bool, optional
            Print out a summary. The default is False.

        Returns
        -------
        run_idx : int
            Run number.
        array
            Optimial candidate.

        """
        if exploit_method.lower() not in ["varoptdesign"]:
            raise AttributeError(
                f"Dual annealing does not support exploitation with {exploit_method}"
            )

        max_func_itr = self.exp_design.max_func_itr

        res_global = None
        if exploit_method.lower() == "varoptdesign":
            self.x_mc = self.exp_design.generate_samples(self.mc_size, "random")
            res_global = opt.dual_annealing(
                self.util_var_opt_design,
                bounds=bounds,
                maxfun=max_func_itr,
            )

        if verbose:
            print(
                f"Global minimum: xmin = {res_global.x}, "
                f"f(xmin) = {res_global.fun:.6f}, nfev = {res_global.nfev}"
            )

        return run_idx, res_global.x

    # -------------------------------------------------------------------------
    def util_alph_opt_design(self, all_candidates):
        """
        Enriches the Experimental design with the requested alphabetic
        criterion based on exploring the space with number of sampling points.

        Ref: Hadigol, M., & Doostan, A. (2018). Least squares polynomial chaos
        expansion: A review of sampling strategies., Computer Methods in
        Applied Mechanics and Engineering, 332, 382-407.

        Arguments
        ---------
        all_candidates : array
            Array with candidate points to be searched

        Returns
        -------
        scores : array of shape (n_candidates,)
            The scores for each candidate point based on the selected utility function.
        """
        # This function currently only supports PCE/aPCE
        if self.meta_model.meta_model_type.lower() not in ["pce", "apce"]:
            raise AttributeError(
                "Alphabetic optimal design currently only support PCE-type models!"
            )
        util_func = self.exp_design.util_func
        n_candidates = all_candidates.shape[0]

        # Old Experimental design
        old_ed_x = self.exp_design.x

        # Suggestion: Go for the output/location with the highest LOO error
        # This is just a patch! (To simplify the optimization, avoid Multi-obj optimization)
        target_loo = -np.inf
        target_out_name = None
        target_out_idx = None

        for key in self.out_names:
            loo_scores = list(self.meta_model.loocv_score_dict["b_1"][key].values())
            mod_loo = [1 - score for score in loo_scores]

            # Find worst location for this output
            local_idx = np.argmax(mod_loo)
            local_loo = mod_loo[local_idx]

            if local_loo > target_loo:
                target_loo = local_loo
                target_out_name = key
                target_out_idx = local_idx

        # Initialize phi to save the criterion's values
        scores = np.zeros(n_candidates)

        # Patch: select the basis indices for the selected output/location with worst loocv
        basis_indices = self.meta_model._basis_dict["b_1"][target_out_name][
            "y_" + str(target_out_idx + 1)
        ]

        # ------ Old Psi ------------
        univ_p_val = self.meta_model.univ_basis_vals(old_ed_x)
        psi = create_psi(basis_indices, univ_p_val)

        # ------ New candidates (Psi_c) ------------
        # Assemble Psi_c
        univ_p_val_c = self.meta_model.univ_basis_vals(all_candidates)
        psi_c = create_psi(basis_indices, univ_p_val_c)

        for idx in range(n_candidates):

            # Include the new row to the original Psi
            psi_cand = np.vstack((psi, psi_c[idx]))

            # Information matrix
            psi_t_psi = np.dot(psi_cand.T, psi_cand)
            m = psi_t_psi / (len(old_ed_x) + 1)

            if 1e-12 < np.linalg.cond(psi_t_psi) < 1 / sys.float_info.epsilon:
                # faster
                inv_m = linalg.solve(m, sparse.eye(psi_t_psi.shape[0]).toarray())
            else:
                # stabler
                inv_m = np.linalg.pinv(m)

            # ---------- Calculate optimality criterion ----------
            # Optimality criteria according to Section 4.5.1 in Ref.

            # D-Opt
            if util_func.lower() == "d-opt":
                scores[idx] = (np.linalg.det(inv_m)) ** (1 / len(basis_indices))

            # A-Opt
            elif util_func.lower() == "a-opt":
                scores[idx] = np.trace(inv_m)

            # K-Opt
            elif util_func.lower() == "k-opt":
                scores[idx] = np.linalg.cond(m)

            else:
                raise AttributeError(
                    "The optimality criterion you requested has "
                    "not been implemented yet!"
                )

        # Set all inf values to nan
        scores[np.isinf(scores)] = np.nan

        if util_func.lower() in ["a-opt", "k-opt"]:
            scores = np.nanmax(scores) - scores  # invert so min becomes max

        # Normalize scores
        norm_score = scores / np.nansum(scores)

        return norm_score

    def _select_indexes(self, prior_samples, collocation_points):
        """
        This function will be used to check the user-input exploration samples,
        remove training points that were already used, and select the first mc_size
        samples that have not yet been used for training. It should also
        assign an exploration score of 0 to all samples.

        Parameters
        ----------
        prior_samples: array [mc_size, n_params]
            Pre-defined samples from the parameter space, out of which the
            sample sets should be extracted.
        collocation_points: [tp_size, n_params]
            array with training points which were already used to train
            the surrogate model, and should therefore
            not be re-explored.

        Returns
        -------
        array[self.mc_size,]
            With indexes of the new candidate parameter sets, to be read from
            the prior_samples array.
        """
        n_tp = collocation_points.shape[0]
        mc_samples = self.exp_design.n_candidates

        # Get index of elements that have already been used
        aux1_ = np.where(
            (prior_samples[: mc_samples + n_tp, :] == collocation_points[:, None]).all(
                -1
            )
        )[1]

        # Give each element in the prior a True if it has not been used before
        aux2_ = np.invert(
            np.in1d(np.arange(prior_samples[: mc_samples + n_tp, :].shape[0]), aux1_)
        )

        # Select the first d_size_bal of the unused elements
        al_unique_index = np.arange(prior_samples[: mc_samples + n_tp, :].shape[0])[
            aux2_
        ]
        al_unique_index = al_unique_index[:mc_samples]

        return al_unique_index
