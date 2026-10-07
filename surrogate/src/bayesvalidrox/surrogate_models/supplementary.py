#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary functions that are used in multiple classes
"""
import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF


# -------------------------------------------------------------------------
def check_ranges(theta, ranges):
    """
    This function checks if theta lies in the given ranges.

    Parameters
    ----------
    theta : array
        Proposed parameter set.
    ranges : nested list
        The parameter ranges.

    Returns
    -------
    c : bool
        If it lies in the given range, it returns True else False.

    """
    c = True
    # traverse in the list1
    for i, bounds in enumerate(ranges):
        x = theta[i]
        # condition check
        if x < bounds[0] or x > bounds[1]:
            c = False
            return c
    return c


# -------------------------------------------------------------------------
def hellinger_distance(p_, q_):
    """
    Hellinger distance between two continuous distributions.

    The maximum distance 1 is achieved when p_ assigns probability zero to
    every set to which q_ assigns a positive probability, and vice versa.
    0 (identical) and 1 (maximally different)

    Parameters
    ----------
    p_ : array
        Reference likelihood.
    q_ : array
        Estimated likelihood.

    Returns
    -------
    float
        Hellinger distance of two distributions.

    """
    p_ = np.array(p_)
    q_ = np.array(q_)

    mu1 = p_.mean()
    sigma1 = np.std(p_)

    mu2 = q_.mean()
    sigma2 = np.std(q_)

    term1 = np.sqrt(2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2))

    term2 = np.exp(-0.25 * (mu1 - mu2) ** 2 / (sigma1**2 + sigma2**2))

    h_squared = 1 - term1 * term2

    return np.sqrt(h_squared)


def subdomain(bounds, n_new_samples):
    """
    Divides a domain defined by Bounds into subdomains.

    Parameters
    ----------
    Bounds : list of tuples
        List of lower and upper bounds.
    n_new_samples : int
        Number of samples to divide the domain for.

    Returns
    -------
    Subdomains : List of tuples of tuples
        Each tuple of tuples divides one set of bounds into n_new_samples parts.

    """
    n_params = len(bounds)
    n_subdomains = n_new_samples + 1
    lin_space = np.zeros((n_params, n_subdomains))

    for i in range(n_params):
        lin_space[i] = np.linspace(
            start=bounds[i][0], stop=bounds[i][1], num=n_subdomains
        )
    subdomains = []
    for k in range(n_subdomains - 1):
        mylist = []
        for i in range(n_params):
            mylist.append((lin_space[i, k + 0], lin_space[i, k + 1]))
        subdomains.append(tuple(mylist))

    return subdomains


# noinspection SpellCheckingInspection
def corr_loocv_error(clf, psi, coeffs, y):
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
    clf : object
        Fitted estimator.
    psi : array of shape (n_samples, n_features)
        The multivariate orthogonal polynomials (regressor).
    coeffs : array-like of shape (n_features,)
        Estimated cofficients.
    y : array of shape (n_samples,)
        Target values.

    Returns
    -------
    r_2 : float
        LOOCV Validation score (1-LOOCV erro).
    residual : array of shape (n_samples,)
        Residual values (y - predicted targets).

    """
    psi = np.array(psi, dtype=float)

    # Create sparse psi matrix by removing redundent terms
    nnz_idx = np.nonzero(coeffs)[0]
    if len(nnz_idx) == 0:
        nnz_idx = [0]
    psi_sparse = psi[:, nnz_idx]

    # NrCoeffs of aPCEs
    p_ = len(nnz_idx)
    # NrEvaluation (Size of experimental design)
    n_ = psi.shape[0]

    # Build the projection matrix
    psi_t_psi = np.dot(psi_sparse.T, psi_sparse)

    if np.linalg.cond(psi_t_psi) > 0:  # and \
        # np.linalg.cond(PsiTPsi) < 1/sys.float_info.epsilon:
        # faster
        try:
            m_ = sp.linalg.solve(psi_t_psi, sp.sparse.eye(psi_t_psi.shape[0]).toarray())
        except Exception as exc:
            raise AttributeError(
                "There are too few samples for the corrected loo-cv error. "
                "Fit surrogate on at least as many "
                "samples as parameters to use this"
            ) from exc
    else:
        # stabler
        m_ = np.linalg.pinv(psi_t_psi)

    # h factor (the full matrix is not calculated explicitly,
    # only the trace is, to save memory)
    psi_m = np.dot(psi_sparse, m_)

    h = np.sum(np.multiply(psi_m, psi_sparse), axis=1, dtype=np.longdouble)  # float128)

    # ------ Calculate Error Loocv for each measurement point ----
    # Residuals
    try:
        residual = clf.predict(psi) - y
    except:
        residual = np.dot(psi, coeffs) - y

    # Variance
    var_y = np.var(y)
    if var_y == 0:
        # norm_emp_error = 0
        loo_error = 0
        lc_error = np.zeros(y.shape)
        return 1 - loo_error, lc_error

    # norm_emp_error = np.mean(residual ** 2) / var_y

    # LCerror = np.divide(residual, (1-h))
    lc_error = residual / (1 - h)
    loo_error = np.mean(np.square(lc_error)) / var_y
    # if there are NaNs, just return an infinite LOO error (this
    # happens, e.g., when a strongly underdetermined problem is solved)
    if np.isnan(loo_error):
        loo_error = np.inf

    # Corrected Error for over-determined system
    tr_m = np.trace(m_)
    if tr_m < 0 or abs(tr_m) > 1e6:
        tr_m = np.trace(np.linalg.pinv(np.dot(psi.T, psi)))

    # Over-determined system of Equation
    if n_ > p_:
        t_factor = n_ / (n_ - p_) * (1 + tr_m)

    # Under-determined system of Equation
    else:
        t_factor = np.inf

    corrected_loo_error = loo_error * t_factor

    r_2 = 1 - corrected_loo_error

    return r_2, lc_error


def create_psi(basis_indices, univ_p_val):
    """
    This function assemble the design matrix Psi from the given basis index
    set INDICES and the univariate polynomial evaluations univ_p_val.

    Parameters
    ----------
    basis_indices : array of shape (n_terms, n_params)
        Multi-indices of multivariate polynomials.
    univ_p_val : array of (n_samples, n_params, n_max+1)
        All univariate regressors up to `n_max`.

    Raises
    ------
    ValueError
        n_terms in arguments do not match.

    Returns
    -------
    psi : array of shape (n_samples, n_terms)
        Multivariate regressors.

    """
    # Check if BasisIndices is a sparse matrix
    sparsity = sp.sparse.issparse(basis_indices)
    if sparsity:
        basis_indices = basis_indices.toarray()

    # Initialization and consistency checks
    # number of input variables
    n_params = univ_p_val.shape[1]

    # Size of the experimental design
    n_samples = univ_p_val.shape[0]

    # number of basis terms
    n_terms = basis_indices.shape[0]

    # check that the variables have consistent sizes
    if n_params != basis_indices.shape[1]:
        raise ValueError(
            f"The shapes of basis_indices ({basis_indices.shape[1]}) and "
            f"univ_p_val ({n_params}) don't match!!"
        )

    # Preallocate the Psi matrix for performance
    psi = np.ones((n_samples, n_terms))
    # Assemble the Psi matrix
    for m in range(basis_indices.shape[1]):
        aa = np.where(basis_indices[:, m] > 0)[0]
        try:
            basis_idx = basis_indices[aa, m]
            bb = univ_p_val[:, m, basis_idx].reshape(psi[:, aa].shape)
            psi[:, aa] = np.multiply(psi[:, aa], bb)
        except ValueError as err:
            raise err
    return psi


# -------------------------------------------------------------------------
def kernel_rbf(x, hyperparameters):
    """
    Isotropic squared exponential kernel.

    Higher l values lead to smoother functions and therefore to coarser
    approximations of the training data. Lower l values make functions
    more wiggly with wide uncertainty regions between training data points.

    sigma_f controls the marginal variance of b(x)

    Parameters
    ----------
    x : ndarray of shape (n_samples_X, n_features)
    hyperparameters : dict
        Lambda characteristic length
        sigma_f controls the marginal variance of b(x)
        sigma_0 unresolvable error nugget term, interpreted as random
                error that cannot be attributed to measurement error.

    Returns
    -------
    var_cov_matrix : ndarray of shape (n_samples_X,n_samples_X)
        Kernel k(X, X).

    """
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(x)

    nparams = len(hyperparameters)
    if nparams < 3:
        raise AttributeError("Provide 3 parameters for the RBF kernel!")

    # characteristic length (0,1]
    lambda_ = hyperparameters[0]
    # sigma_f controls the marginal variance of b(x)
    sigma2_f = hyperparameters[1]

    rbf = RBF(length_scale=lambda_)
    cov_matrix = sigma2_f * rbf(x_minmax)

    # (unresolvable error) nugget term that is interpreted as random
    # error that cannot be attributed to measurement error.
    sigma2_0 = hyperparameters[2:]
    for i, j in np.ndindex(cov_matrix.shape):
        cov_matrix[i, j] += np.sum(sigma2_0) if i == j else 0

    return cov_matrix


# -------------------------------------------------------------------------
def gelman_rubin(chain, return_var=False):
    """
    The potential scale reduction factor (PSRF) defined by the variance
    within one chain, W, with the variance between chains B.
    Both variances are combined in a weighted sum to obtain an estimate of
    the variance of a parameter \\( \\theta \\).The square root of the
    ratio of this estimates variance to the within chain variance is called
    the potential scale reduction.
    For a well converged chain it should approach 1. Values greater than
    1.1 typically indicate that the chains have not yet fully converged.

    Source: http://joergdietrich.github.io/emcee-convergence.html

    https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py

    Parameters
    ----------
    chain : array (n_walkers, n_steps, n_params)
        The emcee ensamples.
    return_var : bool, optioal
        Toggles returning the variance insted of the Gelman-Rubin values.
        The default is False.

    Returns
    -------
    r_hat : float
        The Gelman-Robin values.

    """
    chain = np.array(chain)
    m_chains, n_iters = chain.shape[:2]

    # Calculate between-chain variance
    theta_b = np.mean(chain, axis=1)
    theta_bb = np.mean(theta_b, axis=0)
    b_over_n = ((theta_bb - theta_b) ** 2).sum(axis=0)
    b_over_n /= m_chains - 1

    # Calculate within-chain variances
    ssq = np.var(chain, axis=1, ddof=1)
    w = np.mean(ssq, axis=0)

    # (over) estimate of variance
    var_theta = w * (n_iters - 1) / n_iters + b_over_n

    if return_var:
        return var_theta
    # The square root of the ratio of this estimates variance to the
    # within chain variance
    r_hat = np.sqrt(var_theta / w)
    return r_hat


def root_mean_squared_error(reference, approximation):
    """
    Implementation of RMSE

    Parameters
    ----------
    reference : np.array
        Reference values
    approximation : np.array
        Approximation values

    Returns
    -------
    rmse : np.array
        RMSE of approximation against the reference along the samples.
    """
    rmse = np.sqrt(
        np.mean(np.power(np.array(reference) - np.array(approximation), 2), axis=0)
    )
    return rmse
