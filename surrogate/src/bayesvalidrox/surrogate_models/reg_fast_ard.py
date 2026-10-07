#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class RegressionFastARD, inherits from scikit
"""
import warnings
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve_triangular, pinvh
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_X_y


def update_precisions(q1, s1, q, s, a, active, tol, n_samples, clf_bias):
    """
    Selects one feature to be added/recomputed/deleted to model based on
    effect it will have on value of log marginal likelihood.
    """
    # Initialise vector holding changes in log marginal likelihood
    delta_l = np.zeros(q1.shape[0])

    # Identify features that can be added , recomputed and deleted in model
    theta = q**2 - s
    add = (theta > 0) * (active == False)
    recompute = (theta > 0) * (active == True)
    delete = ~(add + recompute)

    # Compute sparsity & quality parameters corresponding to features in
    # three groups identified above
    q1_add, s1_add = q1[add], s1[add]
    q1_rec, s1_rec, a_rec = q1[recompute], s1[recompute], a[recompute]
    q1_del, s1_del, a_del = q1[delete], s1[delete], a[delete]

    # Compute new alpha's (precision parameters) for features that are
    # currently in model and will be recomputed
    a_new = s[recompute] ** 2 / (theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha = 1.0 / a_new - 1.0 / a_rec

    # Compute change in log marginal likelihood
    delta_l[add] = (q1_add**2 - s1_add) / s1_add + np.log(s1_add / q1_add**2)
    delta_l[recompute] = q1_rec**2 / (s1_rec + 1.0 / delta_alpha) - np.log(
        1 + s1_rec * delta_alpha
    )
    delta_l[delete] = q1_del**2 / (s1_del - a_del) - np.log(1 - s1_del / a_del)
    delta_l = delta_l / n_samples

    # Find feature which caused largest change in likelihood
    feature_index = np.argmax(delta_l)

    # No deletions or additions
    same_features = np.sum(theta[~recompute] > 0) == 0

    # Changes in precision for features already in model is below threshold
    no_delta = np.sum(abs(a_new - a_rec) > tol) == 0

    # Check convergence: if no features to add or delete and small change in
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        return [a, converged]

    # If not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        a[feature_index] = s[feature_index] ** 2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            # do not remove bias term in classification
            # (in regression it is factored in through centering)
            if not (feature_index == 0 and clf_bias):
                active[feature_index] = False
                a[feature_index] = np.inf

    return [a, converged]


class RegressionFastARD(LinearModel, RegressorMixin):
    """
    Regression with Automatic Relevance Determination (Fast Version uses
    Sparse Bayesian Learning)
    https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations

    start: list, optional (DEFAULT = None)
        Initial selected features.

    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_x : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

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
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization.

    References
    ----------
    [1] Fast marginal likelihood maximisation for sparse Bayesian models
    (Tipping & Faul 2003) (http://www.miketipping.com/papers/met-fastsbl.pdf)
    [2] Analysis of sparse Bayesian learning (Tipping & Faul 2001)
        (http://www.miketipping.com/abstracts.htm#Faul:NIPS01)
    """

    def __init__(
        self,
        n_iter=300,
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

        # Other parameters
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

    def _preprocess_data(self, X, y):
        """
        Centers data to have mean zero along axis 0. If fit_intercept=False or
        if the X is a sparse matrix, no centering is done, but normalization
        can still be applied. The function returns the statistics necessary to
        reconstruct the input data, which are x_offset, y_offset, x_scale, such
        that the output
            X = (X - x_offset) / x_scale

        Parameters
        ----------
        X: {array-like, sparse matrix} of size (n_samples, n_features)
           Training data, matrix of explanatory variables
        y: array-like of size [n_samples, n_features]
           Target values

        Returns
        -------
        X: {array-like, sparse matrix}
            Processed training samples.
        y : array-like
            Processed training outputs.
        x_offset : array-like
            Offsets for X
        y_offset : array-like
            Offsets for y
        x_scale : array-like
            L2 norm of X - x_offset.

        """

        if self.copy_x:
            X = X.copy(order="K")

        y = np.asarray(y, dtype=X.dtype)

        if self.fit_intercept:
            x_offset = np.average(X, axis=0)
            X -= x_offset
            if self.normalize:
                x_scale = np.ones(X.shape[1], dtype=X.dtype)
                std = np.sqrt(np.sum(X**2, axis=0) / (len(X) - 1))
                x_scale[std != 0] = std[std != 0]
                X /= x_scale
            else:
                x_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = np.mean(y)
            y = y - y_offset
        else:
            x_offset = np.zeros(X.shape[1], dtype=X.dtype)
            x_scale = np.ones(X.shape[1], dtype=X.dtype)
            if y.ndim == 1:
                y_offset = X.dtype.type(0)
            else:
                y_offset = np.zeros(y.shape[1], dtype=X.dtype)

        return X, y, x_offset, y_offset, x_scale

    def fit(self, X, y):
        """
        Fits ARD Regression with Sequential Sparse Bayes Algorithm.

        Parameters
        -----------
        X: {array-like, sparse matrix} of size (n_samples, n_features)
           Training data, matrix of explanatory variables
        y: array-like of size [n_samples, n_features]
           Target values

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape

        X, y, x_mean, y_mean, x_std = self._preprocess_data(X, y)
        self._x_mean_ = x_mean
        self._y_mean = y_mean
        self._x_std = x_std

        # Precompute X'*Y , X'*X for faster iterations & allocate memory for
        # sparsity & quality vectors
        xy = np.dot(X.T, y)
        xx = np.dot(X.T, X)
        xxd = np.diag(xx)

        # Initialise precision of noise & and coefficients
        var_y = np.var(y)

        # Check that variance is non zero !!!
        if var_y == 0:
            beta = 1e-2
            self.var_y = True
        else:
            beta = 1.0 / np.var(y)
            self.var_y = False

        a = np.inf * np.ones(n_features)
        active = np.zeros(n_features, dtype=bool)

        if self.start is not None and not hasattr(self, "active_"):
            start = self.start
            # Start from a given start basis vector
            proj = xy**2 / xxd
            active[start] = True
            a[start] = xxd[start] / (proj[start] - var_y)

        else:
            # In case of almost perfect multicollinearity between some features
            # start from feature 0
            if np.sum(xxd - x_mean**2 < np.finfo(np.float32).eps) > 0:
                a[0] = np.finfo(np.float16).eps
                active[0] = True

            else:
                # Start from a single basis vector with largest projection on
                # targets
                proj = xy**2 / xxd
                start = np.argmax(proj)
                active[start] = True
                a[start] = xxd[start] / (proj[start] - var_y + np.finfo(np.float32).eps)

        warning_flag = 0
        scores_ = []
        for i in range(self.n_iter):
            # Handle variance zero
            if self.var_y:
                a[0] = y_mean
                active[0] = True
                converged = True
                break

            xxa = xx[active, :][:, active]
            xya = xy[active]
            aa = a[active]

            # Mean & covariance of posterior distribution
            mn, ri, cholesky = self._posterior_dist(aa, beta, xxa, xya)
            if cholesky:
                s_diag = np.sum(ri**2, 0)
            else:
                s_diag = np.copy(np.diag(ri))
                warning_flag += 1

            # Raise warning in case cholesky fails
            if warning_flag == 1:
                warnings.warn(
                    (
                        "Cholesky decomposition failed! Algorithm uses "
                        "pinvh, which is significantly slower. If you "
                        "use RVR it is advised to change parameters of "
                        "the kernel!"
                    )
                )

            # Compute quality & sparsity parameters
            s, q, s1, q1 = self._sparsity_quality(
                xx, xxd, xy, xya, aa, ri, active, beta, cholesky
            )

            # Update precision parameter for noise distribution
            rss = np.sum((y - np.dot(X[:, active], mn)) ** 2)

            # If near perfect fit , then terminate
            if (rss / n_samples / var_y) < self.tol:
                warnings.warn("Early termination due to near perfect fit")
                converged = True
                break
            beta = n_samples - np.sum(active) + np.sum(aa * s_diag)
            beta /= rss
            # beta /= (rss + np.finfo(np.float32).eps)

            # Update precision parameters of coefficients
            a, converged = update_precisions(
                q1, s1, q, s, a, active, self.tol, n_samples, False
            )

            if self.compute_score:
                scores_.append(self.log_marginal_like(xxa, xya, aa, beta))

            if self.verbose:
                print(
                    (
                        f"Iteration: {i}, number of features in the model: {np.sum(active)}"
                    )
                )

            if converged or i == self.n_iter - 1:
                if converged and self.verbose:
                    print("Algorithm converged!")
                break

        # After last update of alpha & beta update parameters
        # of posterior distribution
        xxa, xya, aa = xx[active, :][:, active], xy[active], a[active]
        mn, sn, cholesky = self._posterior_dist(aa, beta, xxa, xya, True)
        self.coef_ = np.zeros(n_features)
        self.coef_[active] = mn
        self.sigma_ = sn
        self.active_ = active
        self.lambda_ = a
        self.alpha_ = beta
        self.converged = converged
        if self.compute_score:
            self.scores_ = np.array(scores_)

        # set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / x_std
            self.intercept_ = y_mean - np.dot(x_mean, self.coef_.T)
        else:
            self.intercept_ = 0.0
        return self
    
    def log_marginal_like(self, xxa, xya, aa, beta):
        """Computes the log of the marginal likelihood."""
        n, _ = xxa.shape
        a = np.diag(aa)

        _, sigma_, _ = self._posterior_dist(aa, beta, xxa, xya, full_covar=True)

        c = sigma_ + np.dot(np.dot(xxa.T, np.linalg.pinv(a)), xxa)

        score = (
            np.dot(np.dot(xya.T, np.linalg.pinv(c)), xya)
            + np.log(np.linalg.det(c))
            + n * np.log(2 * np.pi)
        )

        return -0.5 * score

    def predict(self, X, return_std=False):
        """
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and variance based on
        Ref.[1] Section 3.3.2.

        Parameters
        -----------
        x: {array-like, sparse} (n_samples_test, n_features)
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

            if self.normalize:
                X -= self._x_mean_[self.active_]
                X /= self._x_std[self.active_]
            var_hat = 1.0 / self.alpha_
            var_hat += np.sum(X.dot(self.sigma_) * X, axis=1)
            std_hat = np.sqrt(var_hat)
            return y_hat, std_hat
        return y_hat

    def _posterior_dist(self, a, beta, xx, xy, full_covar=False):
        """
        Calculates mean and covariance matrix of posterior distribution
        of coefficients.
        """
        # Compute precision matrix for active features
        sinv = beta * xx
        np.fill_diagonal(sinv, np.diag(sinv) + a)
        cholesky = True

        # Try cholesky, if it fails go back to pinvh
        try:
            # Find posterior mean : R*R.T*mean = beta*X.T*Y
            # Solve(R*z = beta*X.T*Y) =>find z=> solve(R.T*mean = z)=>find mean
            r = np.linalg.cholesky(sinv)
            z = solve_triangular(r, beta * xy, check_finite=True, lower=True)
            mn = solve_triangular(r.T, z, check_finite=True, lower=False)

            # Invert lower triangular matrix from cholesky decomposition
            ri = solve_triangular(r, np.eye(a.shape[0]), check_finite=False, lower=True)
            if full_covar:
                sn = np.dot(ri.T, ri)
                return mn, sn, cholesky
            return mn, ri, cholesky
        except LinAlgError:
            cholesky = False
            sn = pinvh(sinv)
            mn = beta * np.dot(sinv, xy)
            return mn, sn, cholesky

    def _sparsity_quality(self, xx, xxd, xy, xya, aa, ri, active, beta, cholesky):
        """
        Calculates sparsity and quality parameters for each feature

        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X * sn * X'
        """
        bxy = beta * xy
        bxx = beta * xxd
        if cholesky:
            # Here ri is inverse of lower triangular matrix obtained from
            # cholesky decomp
            xxr = np.dot(xx[:, active], ri.T)
            rxy = np.dot(ri, xya)
            s = bxx - beta**2 * np.sum(xxr**2, axis=1)
            q = bxy - beta**2 * np.dot(xxr, rxy)
        else:
            # Here ri is covariance matrix
            xxa = xx[:, active]
            xs = np.dot(xxa, ri)
            s = bxx - beta**2 * np.sum(xs * xxa, 1)
            q = bxy - beta**2 * np.dot(xs, xya)
        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S)
        # so if A = np.inf q = Q, s = S
        qi = np.copy(q)
        si = np.copy(s)
        # If A is not np.inf, then it should be 'active' feature => use (EQ 1)
        qa, sa = q[active], s[active]
        qi[active] = aa * qa / (aa - sa)
        si[active] = aa * sa / (aa - sa)

        return [si, qi, s, q]
