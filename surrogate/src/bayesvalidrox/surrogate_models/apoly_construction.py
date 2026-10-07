#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construction of polynomials for aPCE
"""
import numpy as np


def apoly_construction(data, degree):
    """
    Construction of data-driven Orthonormal Polynomial Basis
    Author: Dr.-Ing. habil. Sergey Oladyshkin
    Department of Stochastic Simulation and Safety Research for Hydrosystems
    Institute for Modelling Hydraulic and Environmental Systems
    Universitaet Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
    E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
    http://www.iws-ls3.uni-stuttgart.de
    The current script is based on definition of arbitrary polynomial chaos
    expansion (aPC), which is presented in the following manuscript:
    Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using
    the arbitrary polynomial chaos expansion. Reliability Engineering & System
    Safety, Elsevier, V. 106, P.  179-190, 2012.
    DOI: 10.1016/j.ress.2012.05.002.

    Parameters
    ----------
    data : array
        Raw data.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    Polynomial : array
        The coefficients of the univariate orthonormal polynomials.

    """
    if data.ndim != 1:
        raise AttributeError("Data should be a 1D array")

    # Initialization
    dd = degree + 1
    nsamples = len(data)

    # Forward linear transformation (Avoiding numerical issues)
    data_mean = np.mean(data)
    data = data / data_mean

    # Compute raw moments of input data
    raw_moments = [np.sum(np.power(data, p)) / nsamples for p in range(2 * dd + 2)]

    # Main Loop for polynomial with degree up to dd
    polycoeff_nonnorm = np.empty((0, 1))
    polynomial = np.zeros((dd + 1, dd + 1))

    for degree_ in range(dd + 1):
        mm = np.zeros((degree_ + 1, degree_ + 1))
        vc = np.zeros((degree_ + 1))

        # Define Moments Matrix mm
        for i in range(degree_ + 1):
            for j in range(degree_ + 1):
                if i < degree_:
                    mm[i, j] = raw_moments[i + j]

                elif (i == degree_) and (j == degree_):
                    mm[i, j] = 1

            # Numerical Optimization for Matrix Solver
            mm[i] = mm[i] / max(abs(mm[i]))

        # Defenition of Right Hand side ortogonality conditions: vc
        for i in range(degree_ + 1):
            vc[i] = 1 if i == degree_ else 0

        # Solution: Coefficients of Non-Normal Orthogonal Polynomial: vp Eq.(4)
        try:
            vp = np.linalg.solve(mm, vc)
        except:
            inv_mm = np.linalg.pinv(mm)
            vp = np.dot(inv_mm, vc.T)

        if degree_ == 0:
            polycoeff_nonnorm = np.append(polycoeff_nonnorm, vp)

        if degree_ != 0:
            if degree_ == 1:
                zero = [0]
            else:
                zero = np.zeros((degree_, 1))
            polycoeff_nonnorm = np.hstack((polycoeff_nonnorm, zero))

            polycoeff_nonnorm = np.vstack((polycoeff_nonnorm, vp))

        if 100 * abs(sum(abs(np.dot(mm, vp)) - abs(vc))) > 0.5:
            print("\n---> Attention: Computational Error too high !")
            print("\n---> Problem: Convergence of Linear Solver")

        # Original Numerical Normalization of Coefficients with Norm and
        # orthonormal Basis computation Matrix Storrage
        # Note: polynomial(i,j) correspont to coefficient number "j-1"
        # of polynomial degree "i-1"
        p_norm = 0
        for i in range(nsamples):
            poly = 0
            for k in range(degree_ + 1):
                if degree_ == 0:
                    poly += polycoeff_nonnorm[k] * (data[i] ** k)
                else:
                    poly += polycoeff_nonnorm[degree_, k] * (data[i] ** k)

            p_norm += poly**2 / nsamples

        p_norm = np.sqrt(p_norm)

        for k in range(degree_ + 1):
            if degree_ == 0:
                polynomial[degree_, k] = polycoeff_nonnorm[k] / p_norm
            else:
                polynomial[degree_, k] = polycoeff_nonnorm[degree_, k] / p_norm

    # Backward linear transformation to the real data space
    data *= data_mean
    for k in range(len(polynomial)):
        polynomial[:, k] = polynomial[:, k] / (data_mean ** (k))

    return polynomial
