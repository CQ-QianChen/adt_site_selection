#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class OrthogonalMatchingPursuit, inherits from scikit
"""
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_X_y


def corr(x, y):
    """
    Calculate correlation
    """
    return abs(x.dot(y)) / np.sqrt((x**2).sum())


class OrthogonalMatchingPursuit(LinearModel, RegressorMixin):
    """
    Regression with Orthogonal Matching Pursuit [1].

    Parameters
    ----------
    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_x : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional (DEFAULT = FALSE)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    References
    ----------
    [1] Pati, Y., Rezaiifar, R., Krishnaprasad, P. (1993). Orthogonal matching
        pursuit: recursive function approximation with application to wavelet
        decomposition. Proceedings of 27th Asilomar Conference on Signals,
        Systems and Computers, 40-44.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_x=True, verbose=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_x = copy_x
        self.verbose = verbose

        # Other attributes
        self._x_mean = None
        self._x_mean_ = None
        self._y_mean = None
        self._x_std = None
        self.coef_ = None
        self.active = None
        self.intercept_ = None

    def _preprocess_data(self, X, y):
        """Center and scale data.
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
        Fits Regression with Orthogonal Matching Pursuit Algorithm.

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

        # Normalize columns of Psi, so that each column has norm = 1
        norm_x = np.linalg.norm(X, axis=0)
        x_norm = X / norm_x

        # Initialize residual vector to full model response and normalize
        cap_r = y
        norm_y = np.sqrt(np.dot(y, y))
        r = y / norm_y

        # Check for constant regressors
        const_indices = np.where(~np.diff(X, axis=0).any(axis=0))[0]
        has_const_feature = const_indices.size > 0

        # Start regression using OPM algorithm
        precision = 1e-12  # Set precision criterion to precision of program
        early_stop = True
        cond_early = True  # Initialize condition for early stop
        ind = []
        iindx = []  # index of selected columns
        indtot = np.arange(n_features)  # Full index set for remaining columns
        kmax = min(n_samples, n_features)  # Maximum number of iterations
        loo = np.inf * np.ones(kmax)  # Store LOO error at each iteration
        loo_min = np.inf  # Initialize minimum value of LOO
        coeff = np.zeros((n_features, kmax))
        count = 0
        k = 0.1  # Percentage of iteration history for early stop

        # Begin iteration over regressors set (Matrix X)
        while (
            (np.linalg.norm(cap_r) > precision)
            and (count <= kmax - 1)
            and ((cond_early or early_stop) ^ ~cond_early)
        ):

            # Update index set of columns yet to select
            if count != 0:
                indtot = np.delete(indtot, iindx)

            # Find column of X that is most correlated with residual
            h = abs(np.dot(r, x_norm))
            iindx = np.argmax(h[indtot])
            indx = indtot[iindx]

            # initialize with the constant regressor, if it exists in the basis
            if (count == 0) and has_const_feature:
                indx = const_indices[0]

            # Invert the information matrix at the first iteration, later only
            # update its value on the basis of the previously inverted one,
            if count == 0:
                m = 1 / np.dot(X[:, indx], X[:, indx])
            else:
                x = np.dot(X[:, ind].T, X[:, indx])
                r = np.dot(X[:, indx], X[:, indx])
                m = self.blockwise_inverse(m, x, x.T, r)

            # Add newly found index to the selected indexes set
            ind.append(indx)

            # Select regressors subset (Projection subspace)
            x_pro = X[:, ind]

            # Obtain coefficient by performing OLS
            tt = np.dot(y, x_pro)
            beta = np.dot(m, tt)
            coeff[ind, count] = beta

            # Compute LOO error
            loo[count] = self.loo_error(x_pro, m, y, beta)

            # Compute new residual due to new projection
            cap_r = y - np.dot(x_pro, beta)

            # Normalize residual
            norm_r = np.sqrt(np.dot(cap_r, cap_r))
            r = cap_r / norm_r

            # Update counters and early-stop criterions
            countinf = max(0, int(count - k * kmax))
            loo_min = min(loo_min, loo[count])

            if count == 0:
                cond_early = loo[0] <= loo_min
            else:
                cond_early = min(loo[countinf : count + 1]) <= loo_min

            if self.verbose:
                print(f"Iteration: {count+1}, mod. LOOCV error : " f"{loo[count]:.2e}")

            # Update counter
            count += 1

        # Select projection with smallest cross-validation error

        valid_loo = loo[:count]
        if len(valid_loo) == 0 or np.all(np.isinf(valid_loo)):
            self.coef_ = np.zeros(n_features)
            self.active = np.zeros(n_features, dtype=bool)
        else:
            countmin = np.argmin(valid_loo)
            self.coef_ = coeff[:, countmin]
            self.active = coeff[:, countmin] != 0.0
        # Set intercept_
        if self.fit_intercept:
            self.coef_ = self.coef_ / x_std
            self.intercept_ = y_mean - np.dot(x_mean, self.coef_.T)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        """
        Computes predictive distribution for test set.

        Parameters
        -----------
        X: {array-like, sparse} (n_samples_test, n_features)
           Test data, matrix of explanatory variables

        Returns
        -------
        y_hat: numpy array of size (n_samples_test,)
               Estimated values of targets on test set (i.e. mean of
               predictive distribution)
        """

        y_hat = np.dot(X, self.coef_) + self.intercept_

        return y_hat

    def loo_error(self, psi, inv_inf_matrix, y, coeffs):
        """
        Calculates the corrected LOO error for regression on regressor
        matrix `psi` that generated the coefficients based on [1] and [2].

        [1] Blatman, G., 2009. Adaptive sparse polynomial chaos expansions for
            uncertainty propagation and sensitivity analysis (Doctoral
            dissertation, Clermont-Ferrand 2).

        [2] Blatman, G. and Sudret, B., 2011. Adaptive sparse polynomial chaos
            expansion based on least angle regression. Journal of computational
            Physics, 230(6), pp.2345-2367.

        Parameters
        ----------
        psi : array of shape (n_samples, n_feature)
            Orthogonal bases evaluated at the samples.
        inv_inf_matrix : array
            Inverse of the information matrix.
        y : array of shape (n_samples, )
            Targets.
        coeffs : array
            Computed regresssor cofficients.

        Returns
        -------
        loo_error : float
            Modified LOOCV error.

        """

        # NrEvaluation (Size of experimental design)
        n, p = psi.shape

        # h factor (the full matrix is not calculated explicitly,
        # only the trace is, to save memory)
        psi_m = np.dot(psi, inv_inf_matrix)

        h = np.sum(np.multiply(psi_m, psi), axis=1, dtype=np.longdouble)

        # ------ Calculate Error Loocv for each measurement point ----
        # Residuals
        residual = np.dot(psi, coeffs) - y

        # Variance
        var_y = np.var(y)

        if var_y == 0:
            loo_error = 0
        else:
            loo_error = np.mean(np.square(residual / (1 - h))) / var_y

            # If there are NaNs, just return an infinite LOO error (this
            # happens, e.g., when a strongly underdetermined problem is solved)
            if np.isnan(loo_error):
                loo_error = np.inf

        # Corrected Error for over-determined system
        tr_m = np.trace(np.atleast_2d(inv_inf_matrix))
        if tr_m < 0 or abs(tr_m) > 1e6:
            tr_m = np.trace(np.linalg.pinv(np.dot(psi.T, psi)))

        # Over-determined system of Equation
        if n > p:
            t_factor = n / (n - p) * (1 + tr_m)

        # Under-determined system of Equation
        else:
            t_factor = np.inf

        loo_error *= t_factor

        return loo_error

    def blockwise_inverse(self, a_inv, b, c, d):
        """
        Non-singular square matrix M defined as M = [[A B]; [C D]] .
        B, C and D can have any dimension, provided their combination defines
        a square matrix M.

        Parameters
        ----------
        a_inv : float or array
            inverse of the square-submatrix A.
        b : float or array
            Information matrix with all new regressor.
        c : float or array
            Transpose of B.
        d : float or array
            Information matrix with all selected regressors.

        Returns
        -------
        M : array
            Inverse of the information matrix.

        """
        # Schur complement
        if np.isscalar(d):
            sc_inv = 1 / (d - np.dot(c, np.dot(a_inv, b[:, None])))[0]
        else:
            sc_inv = np.linalg.solve((d - c * a_inv * b), np.eye(d.shape))

        t1 = np.dot(a_inv, np.dot(b[:, None], sc_inv))
        t2 = np.dot(c, a_inv)

        # Assemble the inverse matrix
        m = np.vstack(
            (np.hstack((a_inv + t1 * t2, -t1)), np.hstack((-(sc_inv) * t2, sc_inv)))
        )
        return m
