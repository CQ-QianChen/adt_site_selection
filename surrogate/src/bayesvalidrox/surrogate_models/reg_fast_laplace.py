#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of FastLaplace in scikit style.
"""
import numpy as np
from sklearn.utils import as_float_array
from sklearn.model_selection import KFold


class RegressionFastLaplace:
    """
    Sparse regression with Bayesian Compressive Sensing as described in Alg. 1
    (Fast Laplace) of Ref.[1], which updated formulas from [2].

    sigma2: noise precision (sigma^2)
    nu fixed to 0

    uqlab/lib/uq_regression/BCS/uq_bsc.m

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 1000)
        Maximum number of iterations

    tol: float, optional (DEFAULT = 1e-7)
        If absolute change in precision parameter for weights is below
        threshold algorithm terminates.

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_x : boolean, optional (DEFAULT = True)
        If True, xwill be copied; else, it may be overwritten.

    verbose : boolean, optional (DEFAULT = FALSE)
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
        estimated covariance matrixof the weights, computed only
        for non-zero coefficients

    References
    ----------
    [1] Babacan, S. D., Molina, R., & Katsaggelos, A. K. (2009). Bayesian
        compressive sensing using Laplace priors. IEEE Transactions on image
        processing, 19(1), 53-63.
    [2] Fast marginal likelihood maximisation for sparse Bayesian models
        (Tipping & Faul 2003).
        (http://www.miketipping.com/papers/met-fastsbl.pdf)
    """

    def __init__(
        self,
        n_iter=1000,
        n_kfold=10,
        tol=1e-7,
        fit_intercept=False,
        bias_term=True,
        copy_x=True,
        verbose=False,
    ):
        self.n_iter = n_iter
        self.n_kfold = n_kfold
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.bias_term = bias_term
        self.copy_x = copy_x
        self.verbose = verbose

        # Other parameters
        self.kfold_cv_error = None
        self._x_mean_ = None
        self._y_mean = None
        self._x_std = None
        self.var_y = None
        self.coef_ = None
        self.sigma_ = None
        self.active_ = None
        self.gamma = None
        self.lambda_ = None
        self.beta = None
        self.bcs_path = None
        self.intercept_ = None

    def _center_data(self, X, y):
        """Centers data"""
        x = as_float_array(X, copy=self.copy_x)

        # normalisation should be done in preprocessing!
        x_std = np.ones(X.shape[1], dtype=X.dtype)
        if self.fit_intercept:
            x_mean = np.average(X, axis=0)
            y_mean = np.average(y, axis=0)
            x -= x_mean
            y -= y_mean
        else:
            x_mean = np.zeros(X.shape[1], dtype=X.dtype)
            y_mean = 0.0 if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
        return X, y, x_mean, y_mean, x_std

    def fit(self, X, y):
        """
        Performs the regression

        Parameters
        ----------
        x : TYPE
            Inputs to fit to.
        y : TYPE
            Outputs to fit to.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        k_fold = KFold(n_splits=self.n_kfold)

        var_y = np.var(y, ddof=1) if np.var(y, ddof=1) != 0 else 1.0
        sigma2s = len(y) * var_y * (10 ** np.linspace(-16, -1, self.n_kfold))

        errors = np.zeros((len(sigma2s), self.n_kfold))
        for s, sigma2 in enumerate(sigma2s):
            for k, (train, test) in enumerate(k_fold.split(X, y)):
                self.fit_(X[train], y[train], sigma2)
                errors[s, k] = np.linalg.norm(
                    y[test] - self.predict(X[test])
                ) ** 2 / len(test)

        kf_cv_error = np.sum(errors, axis=1) / self.n_kfold / var_y
        i_min_cv = np.argmin(kf_cv_error)

        self.kfold_cv_error = np.min(kf_cv_error)

        return self.fit_(X, y, sigma2s[i_min_cv])

    def fit_(self, X, y, sigma2):
        """
        Performs the regression

        Parameters
        ----------
        x : TYPE
            Inputs to fit to.
        y : TYPE
            Outputs to fit to.
        sigma2 : TYPE


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        _, p = X.shape
        # n_samples, n_features = X.shape

        X, y, x_mean, y_mean, x_std = self._center_data(X, y)
        self._x_mean_ = x_mean
        self._y_mean = y_mean
        self._x_std = x_std

        # check that variance is non zero !!!
        if np.var(y) == 0:
            self.var_y = True
        else:
            self.var_y = False
        beta = 1.0 / sigma2

        #  precompute X'*Y , X'*x for faster iterations & allocate memory for
        #  sparsity & quality vectors X=Psi
        psi_t_y = np.dot(X.T, y)
        psi_t_psi = np.dot(X.T, X)
        xxd = np.diag(psi_t_psi)

        # initialize with constant regressor, or if that one does not exist,
        # with the one that has the largest correlation with Y
        ind_global_to_local = np.zeros(p, dtype=np.int32)

        # identify constant regressors
        constidx = np.where(~np.diff(X, axis=0).all(axis=0))[0]

        if self.bias_term and constidx.size != 0:
            ind_start = constidx[0]
            ind_global_to_local[ind_start] = True
        else:
            # start from a single basis vector with largest projection on
            # targets
            proj = np.divide(np.square(psi_t_y), xxd)
            ind_start = np.argmax(proj)
            ind_global_to_local[ind_start] = True

        num_active = 1
        active_indices = [ind_start]
        deleted_indices = []
        bcs_path = [ind_start]
        gamma = np.zeros(p)
        # for the initial value of gamma(ind_start), use the RVM formula
        #   gamma = (q^2 - s) / (s^2)
        # and the fact that initially s = S = beta*Psi_i'*Psi_i and q = Q =
        # beta*Psi_i'*Y
        gamma[ind_start] = np.square(psi_t_y[ind_start])
        gamma[ind_start] -= sigma2 * psi_t_psi[ind_start, ind_start]
        gamma[ind_start] /= np.square(psi_t_psi[ind_start, ind_start])

        sigma = 1.0 / (beta * psi_t_psi[ind_start, ind_start] + 1.0 / gamma[ind_start])

        mu = sigma * psi_t_y[ind_start] * beta
        tmp1 = beta * psi_t_psi[ind_start]
        s = beta * np.diag(psi_t_psi).T - sigma * np.square(tmp1)
        q = beta * psi_t_y.T - mu * (tmp1)

        tmp2 = np.ones(p)  # lternative computation for the initial s,q
        q0tilde = psi_t_y[ind_start]
        s0tilde = psi_t_psi[ind_start, ind_start]
        tmp2[ind_start] = s0tilde / (q0tilde**2) / beta
        s = np.divide(s, tmp2)
        q = np.divide(q, tmp2)
        lambda_ = 2 * (num_active - 1) / np.sum(gamma)

        delta_l_max = []
        for i in range(self.n_iter):
            # Handle variance zero
            if self.var_y:
                mu = np.mean(y)
                break

            if self.verbose:
                print(f"    lambda = {lambda_}\n")

            # Calculate the potential updated value of each gamma[i]
            if lambda_ == 0.0:  # RVM
                gamma_potential = np.multiply(
                    ((q**2 - s) > lambda_), np.divide(q**2 - s, s**2)
                )
            else:
                a = lambda_ * s**2
                b = s**2 + 2 * lambda_ * s
                c = lambda_ + s - q**2
                gamma_potential = np.multiply(
                    (c < 0),
                    np.divide(-b + np.sqrt(b**2 - 4 * np.multiply(a, c)), 2 * a),
                )

            l_gamma = -np.log(np.absolute(1 + np.multiply(gamma, s)))
            l_gamma += np.divide(np.multiply(q**2, gamma), (1 + np.multiply(gamma, s)))
            l_gamma -= lambda_ * gamma  # omitted the factor 1/2

            # Contribution of each updated gamma(i) to L(gamma)
            l_gamma_potential = -np.log(
                np.absolute(1 + np.multiply(gamma_potential, s))
            )
            l_gamma_potential += np.divide(
                np.multiply(q**2, gamma_potential),
                (1 + np.multiply(gamma_potential, s)),
            )
            # Omitted the factor 1/2
            l_gamma_potential -= lambda_ * gamma_potential

            # Check how L(gamma) would change if we replaced gamma(i) by the
            # updated gamma_potential(i), for each i separately
            delta_l_potential = l_gamma_potential - l_gamma

            # Deleted indices should not be chosen again
            if len(deleted_indices) != 0:
                values = -np.inf * np.ones(len(deleted_indices))
                delta_l_potential[deleted_indices] = values

            delta_l_max.append(np.nanmax(delta_l_potential))
            ind_l_max = np.nanargmax(delta_l_potential)

            # In case there is only 1 regressor in the model and it would now
            # be deleted
            if (
                len(active_indices) == 1
                and ind_l_max == active_indices[0]
                and gamma_potential[ind_l_max] == 0.0
            ):
                delta_l_potential[ind_l_max] = -np.inf
                delta_l_max[i] = np.max(delta_l_potential)
                ind_l_max = np.argmax(delta_l_potential)

            # If L did not change significantly anymore, break
            if (
                delta_l_max[i] <= 0.0
                or (
                    i > 0
                    and all(
                        np.absolute(delta_l_max[i - 1 :]) < sum(delta_l_max) * self.tol
                    )
                )
                or (i > 0 and all(np.diff(bcs_path)[i - 1 :] == 0.0))
            ):
                if self.verbose:
                    print(
                        f"Increase in L: {delta_l_max[i]} (eta = {self.tol})\
                          -- break\n"
                    )
                break

            if self.verbose:
                print(f"    Delta L = {delta_l_max[i]} \n")

            what_changed = int(gamma[ind_l_max] == 0.0)
            what_changed -= int(gamma_potential[ind_l_max] == 0.0)

            if self.verbose:
                if what_changed < 0:
                    print(f"{i+1} - Remove regressor #{ind_l_max+1}..\n")
                elif what_changed == 0:
                    print(f"{i+1} - Recompute regressor #{ind_l_max+1}..\n")
                else:
                    print(f"{i+1} - Add regressor #{ind_l_max+1}..\n")

            # --- Update all quantities ----
            if what_changed == 1:
                # Adding a regressor

                # Update gamma
                gamma[ind_l_max] = gamma_potential[ind_l_max]

                sigma_ii = 1.0 / (1.0 / gamma[ind_l_max] + s[ind_l_max])
                try:
                    x_i = np.matmul(
                        sigma, psi_t_psi[active_indices, ind_l_max].reshape(-1, 1)
                    )
                except ValueError:
                    x_i = sigma * psi_t_psi[active_indices, ind_l_max]
                tmp_1 = -(beta * sigma_ii) * x_i
                sigma = np.vstack(
                    (
                        np.hstack(
                            ((beta**2 * sigma_ii) * np.dot(x_i, x_i.T) + sigma, tmp_1)
                        ),
                        np.append(tmp_1.T, sigma_ii),
                    )
                )
                mu_i = sigma_ii * q[ind_l_max]
                mu = np.vstack((mu - (beta * mu_i) * x_i, mu_i))

                tmp2_1 = psi_t_psi[:, ind_l_max] - beta * np.squeeze(
                    np.matmul(psi_t_psi[:, active_indices], x_i)
                )
                if i == 0:
                    tmp2_1[0] /= 2
                tmp2 = beta * tmp2_1.T
                s = s - sigma_ii * np.square(tmp2)
                q = q - mu_i * tmp2

                num_active += 1
                ind_global_to_local[ind_l_max] = num_active
                active_indices.append(ind_l_max)
                bcs_path.append(ind_l_max)

            elif what_changed == 0:
                # Zero if regressor has not been chosen yet
                if not ind_global_to_local[ind_l_max]:
                    raise AttributeError(
                        f"Cannot recompute index{ind_l_max} -- not yet\
                                    part of the model!"
                    )
                sigma = np.atleast_2d(sigma)
                mu = np.atleast_2d(mu)

                # Update gamma
                gamma_i_new = gamma_potential[ind_l_max]
                gamma_i_old = gamma[ind_l_max]
                gamma[ind_l_max] = gamma_potential[ind_l_max]

                # Index of regressor in sigma
                local_ind = ind_global_to_local[ind_l_max] - 1

                kappa_i = 1.0 / gamma_i_new - 1.0 / gamma_i_old
                kappa_i = 1.0 / kappa_i
                kappa_i += sigma[local_ind, local_ind]
                kappa_i = 1 / kappa_i
                sigma_i_col = sigma[:, local_ind]

                sigma = sigma - kappa_i * (sigma_i_col * sigma_i_col.T)
                mu_i = mu[local_ind]
                mu = mu - (kappa_i * mu_i) * sigma_i_col[:, None]

                tmp1 = (
                    beta
                    * np.dot(sigma_i_col.reshape(1, -1), psi_t_psi[active_indices])[0]
                )
                s = s + kappa_i * np.square(tmp1)
                q = q + (kappa_i * mu_i) * tmp1

                # No change in active_indices or ind_global_to_local
                bcs_path.append(ind_l_max + 0.1)

            elif what_changed == -1:
                gamma[ind_l_max] = 0

                # Index of regressor in sigma
                local_ind = ind_global_to_local[ind_l_max] - 1

                sigma_ii_inv = 1.0 / sigma[local_ind, local_ind]
                sigma_i_col = sigma[:, local_ind]

                sigma = sigma - sigma_ii_inv * (sigma_i_col * sigma_i_col.T)

                sigma = np.delete(
                    np.delete(sigma, local_ind, axis=0), local_ind, axis=1
                )

                mu = mu - (mu[local_ind] * sigma_ii_inv) * sigma_i_col[:, None]
                mu = np.delete(mu, local_ind, axis=0)

                tmp1 = beta * np.dot(sigma_i_col, psi_t_psi[active_indices])
                s = s + sigma_ii_inv * np.square(tmp1)
                q = q + (mu_i * sigma_ii_inv) * tmp1

                num_active -= 1
                ind_global_to_local[ind_l_max] = 0.0
                v = ind_global_to_local[ind_global_to_local > local_ind] - 1
                ind_global_to_local[ind_global_to_local > local_ind] = v

                # Delete active indices
                del active_indices[local_ind]
                deleted_indices.append(ind_l_max)
                bcs_path.append(-ind_l_max)

            # Same for all three cases
            tmp3 = 1 - np.multiply(gamma, s)
            s = np.divide(s, tmp3)
            q = np.divide(q, tmp3)

            # Update lambda_
            lambda_ = 2 * (num_active - 1) / np.sum(gamma)

        # Prepare the result object
        self.coef_ = np.zeros(p)
        self.coef_[active_indices] = np.squeeze(mu)
        self.sigma_ = sigma
        self.active_ = active_indices
        self.gamma = gamma
        self.lambda_ = lambda_
        self.beta = beta
        self.bcs_path = bcs_path

        # Set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / x_std
            self.intercept_ = y_mean - np.dot(x_mean, self.coef_.T)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X, return_std=False):
        """
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and variance based on
        Ref.[1] Section 3.3.2.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        : list of length two [y_hat, var_hat]

             y_hat: numpy array of size (n_samples_test,)
                    Estimated values of targets on test set (i.e. mean of
                    predictive distribution)

                var_hat: numpy array of size (n_samples_test,)
                    Variance of predictive distribution

        References
        ----------
        [1] Bishop, C. M. (2006). Pattern recognition and machine learning.
        springer.
        """
        y_hat = np.dot(X, self.coef_) + self.intercept_

        if return_std:
            # Handle the zero variance case
            if self.var_y:
                return y_hat, np.zeros_like(y_hat)

            var_hat = 1.0 / self.beta
            var_hat += np.sum(X.dot(self.sigma_) * X, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
        return y_hat
