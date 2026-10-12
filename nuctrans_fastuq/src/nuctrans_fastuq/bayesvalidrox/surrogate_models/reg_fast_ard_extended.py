#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class RegressionFastARDExtended, inherits from scikit-learn LinearModel.

Differences from `reg_fast_ard.py`:
-----------------------------------
1. Highly Optimized Rank-1 Updates: Instead of computing a full Cholesky 
   decomposition of the active feature covariance matrix at every iteration 
   (which scales cubically with the number of active features), this implementation 
   uses rank-1 updates for adding, deleting, and re-estimating features.
2. Memory Efficiency: Avoids computing the full cross-product matrix 
   upfront. Instead, the relevant parts are computed and cached dynamically.
3. Robust Convergence: Closely mirrors the Tipping & Faul algorithm 
   from `bayesSparsify_new.m`. It relies on strict thresholds on the evidence 
   maximization rather than a simple variance tolerance (`tol`).

This implementation is mathematically equivalent but scales significantly better 
to high-dimensional PCE problems.
"""
import warnings
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_X_y


class RegressionFastARDExtended(LinearModel, RegressorMixin):
    """
    Regression with Automatic Relevance Determination (Fast Version uses
    Sparse Bayesian Learning with highly optimized Rank-1 updates).

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 1000)
        Maximum number of iterations
    start: list, optional (DEFAULT = None)
        Initial selected features.
    tol: float, optional (DEFAULT = 1e-3)
        Not actively used for convergence in this optimized method. 
        Included for interface compatibility.
    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model.
    normalize : boolean, optional (DEFAULT = False)
        If True, the regressors X will be normalized.
    copy_x : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.
    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration.
    verbose : boolean, optional (DEFAULT = False)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    alpha_ : float
       estimated precision of the noise
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise
    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood
    """

    def __init__(
        self,
        n_iter=1000,
        start=None,
        tol=1e-3,
        fit_intercept=True,
        normalize=False,
        copy_x=True,
        compute_score=False,
        verbose=False,
    ):
        self.n_iter = n_iter
        self.start = start
        self.tol = tol
        self.scores_ = []
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_x = copy_x
        self.compute_score = compute_score
        self.verbose = verbose

        # Model parameters
        self._x_mean_ = None
        self._y_mean = None
        self._x_std = None
        self.var_y = None
        self.coef_ = None
        self.alpha_ = None
        self.sigma_ = None
        self.active_ = None
        self.lambda_ = None
        self.converged = None
        self.intercept_ = None

    def _preprocess_data(self, x, y):
        """Centers and potentially normalizes the training data."""
        if self.copy_x:
            x = x.copy(order="K")

        y = np.asarray(y, dtype=x.dtype)

        if self.fit_intercept:
            x_offset = np.average(x, axis=0)
            x -= x_offset
            if self.normalize:
                x_scale = np.ones(x.shape[1], dtype=x.dtype)
                std = np.sqrt(np.sum(x**2, axis=0) / (len(x) - 1))
                x_scale[std != 0] = std[std != 0]
                x /= x_scale
            else:
                x_scale = np.ones(x.shape[1], dtype=x.dtype)
            y_offset = np.mean(y)
            y = y - y_offset
        else:
            x_offset = np.zeros(x.shape[1], dtype=x.dtype)
            x_scale = np.ones(x.shape[1], dtype=x.dtype)
            y_offset = x.dtype.type(0) if y.ndim == 1 else np.zeros(y.shape[1], dtype=x.dtype)

        return x, y, x_offset, y_offset, x_scale

    def _compute_residual_energy(self, output_energy, cross_product, output_proj, active_idx, posterior_mean):
        """Computes the energy of the residual."""
        sparse_gram = cross_product[active_idx, :]
        active_proj = output_proj[active_idx]
        residual_energy = output_energy - 2 * np.dot(posterior_mean.T, active_proj) + \
            np.dot(posterior_mean.T, np.dot(sparse_gram, posterior_mean))
        return residual_energy

    def _compute_sparse_bayes_statistics_opt(self, psi_mat, cross_product, output_proj, 
                                           output_energy, active_idx, alpha, beta):
        """Computes sparsity and quality parameters (S and Q) iteratively."""
        n_samples = psi_mat.shape[0]
        sparse_gram = cross_product[active_idx, :]
        active_proj = output_proj[active_idx]
        
        posterior_prec = sparse_gram * beta + np.diag(alpha)
        try:
            cholesky_factor = np.linalg.cholesky(posterior_prec)
            inv_cholesky = np.linalg.solve(cholesky_factor, np.eye(len(active_idx)))
            posterior_cov = np.dot(inv_cholesky.T, inv_cholesky)
            posterior_mean = np.dot(posterior_cov, active_proj) * beta
        except np.linalg.LinAlgError:
            # Fallback for ill-conditioned precision matrix
            from scipy.linalg import pinvh
            posterior_cov = pinvh(posterior_prec)
            posterior_mean = np.dot(posterior_cov, active_proj) * beta
            cholesky_factor = np.eye(len(active_idx))
            inv_cholesky = posterior_cov # dummy to continue without failure

        residual_energy = output_energy - 2 * np.dot(posterior_mean.T, active_proj) + \
            np.dot(posterior_mean.T, np.dot(sparse_gram, posterior_mean))
        
        data_likelihood = (n_samples * np.log(beta) - beta * residual_energy) / 2
        
        # Use log det from cholesky or fallback
        if np.array_equal(cholesky_factor, np.eye(len(active_idx))):
            sign, logdet = np.linalg.slogdet(posterior_prec)
            log_det_half = 0.5 * logdet if sign > 0 else 0
        else:
            log_det_half = np.sum(np.log(np.diag(cholesky_factor)))

        log_marginal_like = data_likelihood - np.dot((posterior_mean**2).T, alpha) / 2 + \
            np.sum(np.log(alpha)) / 2 - log_det_half
            
        gamma = 1 - alpha * np.diag(posterior_cov)
        beta_proj = beta * cross_product
        
        cross_cov = np.dot(beta_proj, posterior_cov)
        # Sum of elementwise multiplication is equivalent to diag(A * B^T)
        s_in = beta - np.sum(cross_cov * beta_proj, axis=1)
        q_in = beta * (output_proj - np.dot(cross_product, posterior_mean))
        
        s_out = np.copy(s_in)
        q_out = np.copy(q_in)
        
        active_scale = alpha / (alpha - s_in[active_idx])
        s_out[active_idx] = active_scale * s_in[active_idx]
        q_out[active_idx] = active_scale * q_in[active_idx]
        theta = q_out**2 - s_out
        
        return (posterior_cov, posterior_mean, s_in, q_in, 
                s_out, q_out, theta, log_marginal_like, 
                gamma, beta_proj)

    def fit(self, x, y):
        """Fits ARD Regression with highly optimized Rank-1 updates."""
        x, y = check_X_y(x, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features_full = x.shape

        x, y, x_mean, y_mean, x_std = self._preprocess_data(x, y)
        self._x_mean_ = x_mean
        self._y_mean = y_mean
        self._x_std = x_std

        var_y = np.var(y)
        if var_y == 0:
            self.var_y = True
            self.coef_ = np.zeros(n_features_full)
            self.intercept_ = y_mean
            self.active_ = np.zeros(n_features_full, dtype=bool)
            self.active_[0] = True
            self.sigma_ = np.zeros((1, 1))
            self.lambda_ = np.inf * np.ones(n_features_full)
            self.alpha_ = 1e-2
            self.converged = True
            return self
            
        self.var_y = False

        # Configuration constants mimicking Tipping & Faul
        zero_factor = 1e-12
        min_delta_log_alpha = 1e-3
        min_delta_log_beta = 1e-6
        beta_update_start = 1
        beta_update_freq = 1
        max_beta_factor = 1e6
        alignment_max = 1 - 1e-3
        use_alignment_test = True
        
        # Action IDs
        ACTION_REESTIMATE = 0
        ACTION_ADD = 1
        ACTION_DELETE = -1
        ACTION_TERMINATE = 10
        ACTION_NOISE_ONLY = 11
        ACTION_ALIGNMENT_SKIP = 12

        # Preprocessing of the polynomial matrix
        x_scales = np.sqrt(np.sum(x**2, axis=0))
        x_scales[x_scales == 0] = 1.0
        psi_mat = x / x_scales[None, :]

        n_samples, n_features = psi_mat.shape
        output_proj = np.dot(psi_mat.T, y)
        output_energy = np.dot(y.T, y)
        max_beta = max_beta_factor / np.var(y)

        # Initialize the first active term
        snr = 0.1
        initial_alpha_max = 1e3
        std_y = max(1e-6, np.std(y))
        beta = 1.0 / (std_y * snr)**2

        if self.start is not None:
            active_idx = [self.start]
        else:
            active_idx = [np.argmax(np.abs(output_proj))]

        sparse_x = psi_mat[:, active_idx]
        scaled_power = beta * np.dot(sparse_x.T, sparse_x)[0, 0]
        scaled_proj = beta * output_proj[active_idx[0]]
        
        alpha = np.zeros(1)
        alpha[0] = scaled_power**2 / (scaled_proj**2 - scaled_power)
        if alpha[0] < 0:
            alpha[0] = initial_alpha_max

        cross_product = np.dot(psi_mat.T, sparse_x)
        
        (posterior_cov, posterior_mean, s_in, q_in,
         s_out, q_out, theta, log_marginal_like,
         gamma, beta_proj) = self._compute_sparse_bayes_statistics_opt(
             psi_mat, cross_product, output_proj,
             output_energy, active_idx, alpha, beta)

        aligned_out = []
        aligned_in = []
        
        scores_ = []

        for i in range(1, self.n_iter + 1):
            n_active = len(active_idx)
            selected_action = ACTION_TERMINATE
            selected_term = 0
            delta_log_marginal = 0.0

            active_theta = theta[active_idx]
            can_reestimate = active_theta > zero_factor
            can_delete = ~can_reestimate
            delete_idx = np.array(active_idx)[can_delete]
            any_can_delete = (len(delete_idx) > 0) and (n_active > 1)

            if any_can_delete:
                # STEP 1: Pruning
                # If any active feature has a relevance factor (theta) <= 0, it should be deleted.
                # We compute the change in log marginal likelihood if we were to delete each candidate,
                # and pick the one that maximizes the evidence.
                delete_delta_lm = -(q_out[delete_idx]**2 /
                    (s_out[delete_idx] + alpha[can_delete]) -
                    np.log(1 + s_out[delete_idx] / alpha[can_delete])) / 2.0
                
                best_del_pos = np.argmax(delete_delta_lm)
                best_del_delta = delete_delta_lm[best_del_pos]
                if best_del_delta > 0:
                    delta_log_marginal = best_del_delta
                    selected_term = delete_idx[best_del_pos]
                    selected_action = ACTION_DELETE

            if not any_can_delete:
                # STEP 2: Adding or Re-estimating
                # If no features need to be deleted, evaluate adding a new candidate feature 
                # or re-estimating the precision (alpha) of an existing active feature.
                delta_lm = np.zeros(n_features) - np.inf
                
                # 2a. Evaluate re-estimating existing active features
                reestimate_idx = np.array(active_idx)[can_reestimate]
                if len(reestimate_idx) > 0:
                    cand_alpha = s_out[reestimate_idx]**2 / theta[reestimate_idx]
                    delta_inv_alpha = (1.0 / cand_alpha) - (1.0 / alpha[can_reestimate])
                    
                    with np.errstate(invalid='ignore', divide='ignore'):
                        dl = (delta_inv_alpha * (q_in[reestimate_idx]**2) /
                              (delta_inv_alpha * s_in[reestimate_idx] + 1) -
                              np.log(1 + s_in[reestimate_idx] * delta_inv_alpha)) / 2.0
                    dl[np.isnan(dl)] = -np.inf
                    delta_lm[reestimate_idx] = dl

                good_theta = theta > zero_factor
                good_theta[active_idx] = False
                if use_alignment_test and len(aligned_out) > 0:
                    good_theta[aligned_out] = False
                
                # 2b. Evaluate adding new features from the candidate pool
                add_idx = np.where(good_theta)[0]
                if len(add_idx) > 0:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        q_s_ratio = q_in[add_idx]**2 / s_in[add_idx]
                        dl = (q_s_ratio - 1 - np.log(q_s_ratio)) / 2.0
                    dl[np.isnan(dl)] = -np.inf
                    delta_lm[add_idx] = dl

                selected_term = np.argmax(delta_lm)
                delta_log_marginal = delta_lm[selected_term]
                
                if delta_log_marginal > 0:
                    if selected_term in active_idx:
                        selected_action = ACTION_REESTIMATE
                    else:
                        selected_action = ACTION_ADD

            selected_active_pos = None
            if selected_action in [ACTION_REESTIMATE, ACTION_DELETE]:
                selected_active_pos = active_idx.index(selected_term)

            selected_alpha = None
            if selected_action in [ACTION_REESTIMATE, ACTION_ADD]:
                selected_alpha = s_out[selected_term]**2 / theta[selected_term]

            if selected_action == ACTION_REESTIMATE:
                prev_alpha = alpha[selected_active_pos]
                if abs(np.log(selected_alpha) - np.log(prev_alpha)) < min_delta_log_alpha:
                    selected_action = ACTION_TERMINATE

            if use_alignment_test:
                # STEP 3: Collinearity Check
                # To maintain numerical stability and well-conditioned matrices, we skip adding 
                # features that are highly collinear (aligned) with already active features.
                if selected_action == ACTION_ADD:
                    selected_x = psi_mat[:, selected_term]
                    alignment = np.dot(selected_x.T, sparse_x)
                    aligned_active_pos = np.where(alignment > alignment_max)[0]
                    if len(aligned_active_pos) > 0:
                        selected_action = ACTION_ALIGNMENT_SKIP
                        aligned_out.extend([selected_term] * len(aligned_active_pos))
                        aligned_in.extend([active_idx[idx] for idx in aligned_active_pos])
                elif selected_action == ACTION_DELETE:
                    delete_align = np.array(aligned_in) == selected_term
                    if np.any(delete_align):
                        aligned_in = np.array(aligned_in)[~delete_align].tolist()
                        aligned_out = np.array(aligned_out)[~delete_align].tolist()

            update_required = False
            new_posterior_cov = posterior_cov

            if selected_action == ACTION_REESTIMATE:
                # --- Rank-1 Update for Re-estimation ---
                # Modifies the covariance matrix and posterior mean when the precision (alpha) 
                # of an already active feature is updated, avoiding a full matrix inversion.
                old_alpha = alpha[selected_active_pos]
                alpha[selected_active_pos] = selected_alpha
                cov_col = posterior_cov[:, selected_active_pos]
                delta_inv_alpha = 1.0 / (selected_alpha - old_alpha)
                kappa = 1.0 / (posterior_cov[selected_active_pos, selected_active_pos] + delta_inv_alpha)
                cov_step = kappa * cov_col
                new_posterior_cov = posterior_cov - np.outer(cov_step, cov_col)
                mean_change = -posterior_mean[selected_active_pos] * cov_step
                posterior_mean = posterior_mean + mean_change
                proj_step = np.dot(beta_proj, cov_col)
                s_in = s_in + kappa * (proj_step**2)
                q_in = q_in - np.dot(beta_proj, mean_change)
                update_required = True

            elif selected_action == ACTION_ADD:
                # --- Rank-1 Update for Addition ---
                # Expands the active sparse polynomial basis by one term.
                # Efficiently updates the covariance matrix from size (KxK) to ((K+1)x(K+1)).
                selected_x = psi_mat[:, selected_term]
                new_proj = np.dot(psi_mat.T, selected_x).reshape(-1, 1)
                cross_product = np.hstack([cross_product, new_proj])
                beta_selected_x = beta * selected_x
                beta_new_proj = beta * new_proj.flatten()
                cross_cov = np.dot(np.dot(beta_selected_x.T, sparse_x), posterior_cov).flatten()
                alpha = np.append(alpha, selected_alpha)
                sparse_x = np.hstack([sparse_x, selected_x.reshape(-1, 1)])
                new_var = 1.0 / (selected_alpha + s_in[selected_term])
                new_cov_col = -new_var * cross_cov
                new_posterior_cov = np.vstack([
                    np.hstack([posterior_cov - new_var * np.outer(cross_cov, cross_cov), new_cov_col.reshape(-1, 1)]),
                    np.append(new_cov_col, new_var)
                ])
                new_mean = new_var * q_in[selected_term]
                mean_change = np.append(-new_mean * cross_cov, new_mean)
                posterior_mean = np.append(posterior_mean, 0) + mean_change
                proj_residual = beta_new_proj - np.dot(beta_proj, cross_cov)
                s_in = s_in - new_var * (proj_residual**2)
                q_in = q_in - new_mean * proj_residual
                active_idx.append(selected_term)
                update_required = True

            elif selected_action == ACTION_DELETE:
                # --- Rank-1 Update for Deletion ---
                # Shrinks the active sparse polynomial basis by one term.
                # Efficiently updates the covariance matrix from size (KxK) to ((K-1)x(K-1)).
                cross_product = np.delete(cross_product, selected_active_pos, axis=1)
                sparse_x = np.delete(sparse_x, selected_active_pos, axis=1)
                alpha = np.delete(alpha, selected_active_pos)
                rem_var = posterior_cov[selected_active_pos, selected_active_pos]
                rem_cov_col = posterior_cov[:, selected_active_pos]
                cov_scale = rem_cov_col / rem_var
                new_posterior_cov = posterior_cov - np.outer(cov_scale, rem_cov_col)
                new_posterior_cov = np.delete(np.delete(new_posterior_cov, selected_active_pos, axis=0), selected_active_pos, axis=1)
                rem_mean = posterior_mean[selected_active_pos]
                mean_change = -rem_mean * cov_scale
                posterior_mean = posterior_mean + mean_change
                posterior_mean = np.delete(posterior_mean, selected_active_pos)
                proj_col = np.dot(beta_proj, rem_cov_col)
                s_in = s_in + (proj_col**2) / rem_var
                q_in = q_in + proj_col * rem_mean / rem_var
                active_idx.pop(selected_active_pos)
                update_required = True

            if update_required:
                # STEP 4: Update Global Statistics
                # After a rank-1 update (Add, Delete, or Re-estimate), we must update 
                # the sparsity (s) and quality (q) factors for all candidate features.
                s_out = np.copy(s_in)
                q_out = np.copy(q_in)
                active_scale = alpha / (alpha - s_in[active_idx])
                s_out[active_idx] = active_scale * s_in[active_idx]
                q_out[active_idx] = active_scale * q_in[active_idx]
                theta = q_out**2 - s_out
                posterior_cov = new_posterior_cov
                gamma = 1 - alpha * np.diag(posterior_cov)
                beta_proj = beta * cross_product
                log_marginal_like = log_marginal_like + delta_log_marginal

            if selected_action == ACTION_TERMINATE or i <= beta_update_start or i % beta_update_freq == 0:
                # STEP 5: Noise Precision (Beta) Update
                # Periodically re-estimate the noise variance (beta) based on the current model residual.
                old_beta = beta
                residual_energy = self._compute_residual_energy(output_energy, cross_product, 
                                                                output_proj, active_idx, posterior_mean)
                beta = (n_samples - np.sum(gamma)) / max(residual_energy, 1e-12)
                beta = min(beta, max_beta)
                delta_log_beta = np.log(beta) - np.log(old_beta)
                if abs(delta_log_beta) > min_delta_log_beta:
                    (posterior_cov, posterior_mean, s_in, q_in,
                     s_out, q_out, theta, log_marginal_like,
                     gamma, beta_proj) = self._compute_sparse_bayes_statistics_opt(
                         psi_mat, cross_product, output_proj,
                         output_energy, active_idx, alpha, beta)
                    if selected_action == ACTION_TERMINATE:
                        selected_action = ACTION_NOISE_ONLY

            if self.compute_score:
                scores_.append(log_marginal_like)

            if selected_action == ACTION_TERMINATE:
                self.converged = True
                if self.verbose:
                    print("Algorithm converged!")
                break
                
            if self.verbose and i % max(1, round(self.n_iter / 10)) == 0:
                print(f"=> Sparse Bayesian Learning: {int((i / self.n_iter) * 100)}% completed")

        self.converged = (selected_action == ACTION_TERMINATE)

        # Back-transform coefficients to the original scales
        sparse_id = np.array(active_idx)
        sort_idx = np.argsort(sparse_id)
        sparse_id = sparse_id[sort_idx]
        
        sparse_coefs = posterior_mean[sort_idx] / x_scales[sparse_id]
        alpha_out = alpha[sort_idx] / (x_scales[sparse_id]**2)

        self.coef_ = np.zeros(n_features_full)
        self.coef_[sparse_id] = sparse_coefs
        
        self.sigma_ = posterior_cov[np.ix_(sort_idx, sort_idx)]
        # Correctly scale sigma_ to original dimensions:
        scaling_matrix = np.outer(1.0 / x_scales[sparse_id], 1.0 / x_scales[sparse_id])
        self.sigma_ = self.sigma_ * scaling_matrix
        
        active_bool = np.zeros(n_features_full, dtype=bool)
        active_bool[sparse_id] = True
        self.active_ = active_bool
        
        self.lambda_ = np.inf * np.ones(n_features_full)
        self.lambda_[sparse_id] = alpha_out
        
        self.alpha_ = beta
        
        if self.compute_score:
            self.scores_ = np.array(scores_)

        if self.fit_intercept:
            self.coef_ = self.coef_ / x_std
            self.intercept_ = y_mean - np.dot(x_mean, self.coef_.T)
        else:
            self.intercept_ = 0.0
            
        return self

    def predict(self, x, return_std=False):
        """Computes predictive distribution for test set."""
        y_hat = np.dot(x, self.coef_) + self.intercept_

        if return_std:
            if self.var_y:
                return y_hat, np.zeros_like(y_hat)

            if self.normalize:
                x_scaled = x - self._x_mean_[self.active_]
                x_scaled = x_scaled / self._x_std[self.active_]
            else:
                x_scaled = x

            var_hat = 1.0 / self.alpha_
            var_hat += np.sum(np.dot(x_scaled, self.sigma_) * x_scaled, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
            
        return y_hat
