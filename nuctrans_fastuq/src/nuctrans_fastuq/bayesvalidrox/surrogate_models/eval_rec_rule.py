#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Based on the implementation in UQLab [1].

References:
1. S. Marelli, and B. Sudret, UQLab: A framework for uncertainty quantification
in Matlab, Proc. 2nd Int. Conf. on Vulnerability, Risk Analysis and Management
(ICVRAM2014), Liverpool, United Kingdom, 2014, 2554-2563.

2. S. Marelli, N. Lüthen, B. Sudret, UQLab user manual – Polynomial chaos
expansions, Report # UQLab-V1.4-104, Chair of Risk, Safety and Uncertainty
Quantification, ETH Zurich, Switzerland, 2021.

Author: Farid Mohammadi, M.Sc.
E-Mail: farid.mohammadi@iws.uni-stuttgart.de
Department of Hydromechanics and Modelling of Hydrosystems (LH2)
Institute for Modelling Hydraulic and Environmental Systems (IWS), University
of Stuttgart, www.iws.uni-stuttgart.de/lh2/
Pfaffenwaldring 61
70569 Stuttgart

Created on Fri Jan 14 2022
"""
import numpy as np
from numpy.polynomial.polynomial import polyval


def poly_rec_coeffs(n_max, poly_type, params=None):
    """
    Computes the recurrence coefficients for classical Wiener-Askey orthogonal
    polynomials.

    Parameters
    ----------
    n_max : int
        Maximum polynomial degree.
    poly_type : string
        Polynomial type.
    params : list, optional
        Parameters required for `laguerre` poly type. The default is None.

    Returns
    -------
    ab : dict
        The 3 term recursive coefficients and the applicable ranges.

    """
    bounds = []
    if poly_type == "legendre":

        def an(n):
            return np.zeros((n + 1, 1))

        def sqrt_bn(n):
            sq_bn = np.zeros((n + 1, 1))
            sq_bn[0, 0] = 1
            for i in range(1, n + 1):
                sq_bn[i, 0] = np.sqrt(1.0 / (4 - i**-2))
            return sq_bn

        bounds = [-1, 1]

    elif poly_type == "hermite":

        def an(n):
            return np.zeros((n + 1, 1))

        def sqrt_bn(n):
            sq_bn = np.zeros((n + 1, 1))
            sq_bn[0, 0] = 1
            for i in range(1, n + 1):
                sq_bn[i, 0] = np.sqrt(i)
            return sq_bn

        bounds = [-np.inf, np.inf]

    elif poly_type == "laguerre":

        def an(n):
            a = np.zeros((n + 1, 1))
            for i in range(1, n + 1):
                a[i] = 2 * n + params[1]
            return a

        def sqrt_bn(n):
            sq_bn = np.zeros((n + 1, 1))
            sq_bn[0, 0] = 1
            for i in range(1, n + 1):
                sq_bn[i, 0] = -np.sqrt(i * (i + params[1] - 1))
            return sq_bn

        bounds = [0, np.inf]

    ab = {
        "alpha_beta": np.concatenate((an(n_max), sqrt_bn(n_max)), axis=1),
        "bounds": bounds,
    }

    return ab


def eval_rec_rule(x, max_deg, poly_type):
    """
    Evaluates the polynomial that corresponds to the Jacobi matrix defined
    from the ab.

    Parameters
    ----------
    x : array (n_samples)
        Points where the polynomials are evaluated.
    max_deg : int
        Maximum degree.
    poly_type : string
        Polynomial type.

    Returns
    -------
    values : array of shape (n_samples, max_deg+1)
        Polynomials corresponding to the Jacobi matrix.

    """
    ab = poly_rec_coeffs(max_deg, poly_type)
    ab = ab["alpha_beta"]

    values = np.zeros((len(x), ab.shape[0] + 1))
    values[:, 1] = 1 / ab[0, 1]

    for k in range(ab.shape[0] - 1):
        values[:, k + 2] = np.multiply((x - ab[k, 0]), values[:, k + 1]) - np.multiply(
            values[:, k], ab[k, 1]
        )
        values[:, k + 2] = np.divide(values[:, k + 2], ab[k + 1, 1])
    return values[:, 1:]


def eval_rec_rule_arbitrary(x, max_deg, poly_coeffs):
    """
    Evaluates the polynomial at sample array x.

    Parameters
    ----------
    x : array (n_samples)
        Points where the polynomials are evaluated.
    max_deg : int
        Maximum degree.
    poly_coeffs : dict
        Polynomial coefficients computed based on moments.

    Returns
    -------
    values : array of shape (n_samples, max_deg+1)
        Univariate Polynomials evaluated at samples.

    """
    values = np.zeros((len(x), max_deg + 1))

    for deg in range(max_deg + 1):
        values[:, deg] = polyval(x, poly_coeffs[deg]).T

    return values


def eval_univ_basis(x, max_deg, poly_types, apoly_coeffs=None):
    """
    Evaluates univariate regressors along input directions.

    Parameters
    ----------
    x : array of shape (n_samples, n_params)
        Training samples.
    max_deg : int
        Maximum polynomial degree.
    poly_types : list of strings
        List of polynomial types for all parameters.
    apoly_coeffs : dict , optional
        Polynomial coefficients computed based on moments. The default is None.

    Returns
    -------
    univ_vals : array of shape (n_samples, n_params, max_deg+1)
        Univariate polynomials for all degrees and parameters evaluated at x.

    """
    # Initilize the output array
    n_samples, n_params = x.shape
    univ_vals = np.zeros((n_samples, n_params, max_deg + 1))

    for i in range(n_params):

        if poly_types[i] == "arbitrary":
            polycoeffs = apoly_coeffs[f"p_{i+1}"]
            univ_vals[:, i] = eval_rec_rule_arbitrary(x[:, i], max_deg, polycoeffs)
        else:
            univ_vals[:, i] = eval_rec_rule(x[:, i], max_deg, poly_types[i])

    return univ_vals
