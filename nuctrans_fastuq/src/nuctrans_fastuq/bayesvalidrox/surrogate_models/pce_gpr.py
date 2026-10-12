#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of metamodel as combination of PC + GPE. PCE can be of type PC or aPC.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .meta_model import (
    MetaModel,
    _preprocessing_fit,
    _preprocessing_eval,
)
from .polynomial_chaos import PCE
from .gaussian_process_sklearn import GPESkl

warnings.filterwarnings("ignore")
# Load the mplstyle
# noinspection SpellCheckingInspection
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


class PCEGPR(MetaModel):
    """
    PCE+GPE MetaModel

    This class trains a surrogate model which is a combination of the types Polynomial Chaos
    Expansion and Gaussian Process Regression. The PCE acts as a de-trender and the GPR is
    trained on the residuals between the PCE and the training data.
    It accepts an input object (input_obj) containing the specification of the distributions
    for uncertain parametersand a model object with instructions on how to run the computational
    model.

    Attributes
    ----------
    input_obj : obj
        Input object with the information on the model input parameters.
    meta_model_type : str
        Surrogate model types (main surrogate). Two surrogate model types are supported:
        polynomial chaos expansion (`PCE_GPE`), arbitrary PCE (`aPCE-GPE`).
        Default is `PCE_GPE`.
    pce_model_type: str
        PCE-surrogate model types (main surrogate). Two surrogate model types are supported:
        polynomial chaos expansion (`PCE`), arbitrary PCE (`aPCE`).
        Default is PCE.
    pce_reg_method : str
        PCE regression method to compute the coefficients. The following
        regression methods are available:
        1. OLS: Ordinary Least Square method
        2. BRR: Bayesian Ridge Regression
        3. LARS: Least angle regression
        4. ARD: Bayesian ARD Regression
        5. FastARD: Fast Bayesian ARD Regression
        6. VBL: Variational Bayesian Learning
        7. EBL: Emperical Bayesian Learning
        Default is `OLS`.
    bootstrap_method : str
        Bootstraping method. Options are `'normal'` and `'fast'`. The default
        is `'fast'`. It means that in each iteration except the first one, only
        the coefficent are recalculated with the ordinary least square method.
    n_bootstrap_itrs : int
        Number of iterations for the bootstrap sampling. The default is `1`.
    pce_deg : int or list of int
        Polynomial degree(s). If a list is given, an adaptive algorithm is used
        to find the best degree with the lowest Leave-One-Out cross-validation
        (LOO) error (or the highest score=1-LOO). Default is `1`.
    pce_q_norm : float
        Hyperbolic (or q-norm) truncation for multi-indices of multivariate
        polynomials. Default is `1.0`.
    kernel_type: str
        Type of kernel to use and train for. The following Scikit-Learn kernels are available:
        1. RBF: Squared exponential kernel
        2. Matern: Matern kernel
        3. RQ: rational quadratic kernel
        Default is `'RBF'` kernel.
    auto_select: bool
        Flag to loop through different available Kernels and select the best one based on BME
        criteria for GPRs.
        Default is False.
    normalize_x_method: str or None
        Type of transformation to apply to the inputs, for the GP training. If None, no
        transformation is done.
        The followint options are available:
        1. 'norm': normalizes inputs U[0,1]
        2. 'standard': standarizes inputs N[0, 1]
        3. 'none': no transformation is done
        4. None: No transformation is done.
    dim_red_method : str
        Dimensionality reduction method for the output space. The available
        method is based on principal component analysis (PCA). The Default is
        `'no'`. There are two ways to select number of components: use
        percentage of the explainable variance threshold (between 0 and 100)
        (Option A) or direct prescription of components' number (Option B):
            >>> MetaModelOpts = MetaModel()
            >>> MetaModelOpts.dim_red_method = 'PCA'
            >>> MetaModelOpts.var_pca_threshold = 99.999  # Option A
            >>> MetaModelOpts.n_pca_components = 12 # Option B
    verbose : bool
        Prints summary of the regression results. Default is `False`.
    """

    def __init__(
        self,
        input_obj,
        meta_model_type="PCE_GPR",
        pce_model_type="PCE",
        pce_reg_method="OLS",
        bootstrap_method="fast",
        n_bootstrap_itrs=1,
        pce_deg=1,
        pce_q_norm=1.0,
        kernel_type="RBF",
        auto_select=True,
        normalize_x_method="norm",
        dim_red_method="no",
        verbose=False,
        n_mc=1000,
        input_transform="user",
    ):

        # Check if the surrogate outputs gaussian results: always True
        is_gaussian = self.check_is_gaussian()

        # Use parent init
        super().__init__(
            input_obj,
            meta_model_type=meta_model_type,
            bootstrap_method=bootstrap_method,
            n_bootstrap_itrs=n_bootstrap_itrs,
            dim_red_method=dim_red_method,
            is_gaussian=is_gaussian,
            verbose=verbose,
            n_mc=n_mc,
            input_transform=input_transform,
        )

        # Additional inputs
        # Parameters that are not needed from the outside are denoted with '_'
        # PCE-related
        self.pce = None
        self.pce_model_type = pce_model_type
        self.pce_reg_method = pce_reg_method
        self.pce_deg = pce_deg
        self.pce_q_norm = pce_q_norm
        self.regression_dict = {}
        self._pce_reg_options = {}

        # GPE related:
        self.gpr = None
        self._kernel_type = kernel_type
        self._auto_select = auto_select
        self.normalize_x_method = normalize_x_method

    def check_is_gaussian(self) -> bool:
        """
        Check if the metamodel returns a mean and stdev. Always true for PCE+GPE

        Returns
        -------
        bool
            DESCRIPTION.

        """
        return True

    def build_metamodel(self) -> None:
        """
        Builds the parts for the metamodel that are needed before fitting.
        Builds the individual PCE and GPR instances
        This is executed outside any loops related to e.g. bootstrap or
        transformations such as pca.

        Returns
        -------
        None

        """
        # Generate PCE ..................................................
        self.pce = PCE(
            input_obj=self.input_obj,
            meta_model_type=self.pce_model_type,
            pce_reg_method=self.pce_reg_method,
            pce_deg=self.pce_deg,
            pce_q_norm=self.pce_q_norm,
        )

        # Generate GPR ....................................................
        self.gpr = GPESkl(
            input_obj=self.input_obj,
            n_restarts=10,  # keep constant
            auto_select=self._auto_select,
            kernel_type=self._kernel_type,
            isotropy=True,  # keep constant
            noisy=True,  # keep constant
            normalize_x_method=self.normalize_x_method,
            verbose=self.verbose,
        )

    @_preprocessing_fit
    def fit(self, X: np.array, y: dict, parallel=False, verbose=False, b_i=0):
        """
        Fits the surrogate to the given data (samples X, outputs y).
        Note here that the samples X should be the transformed samples provided
        by the experimental design if the transformation is used there.

        Parameters
        ----------
        X: 2D list or np.array of shape (#samples, #dim)
            The parameter value combinations that the model was evaluated at.
        y: dict of 2D lists or arrays of shape (#samples, #outputs)
            The respective model evaluations.
        parallel: bool
            Set to True to run the training in parallel for various keys.
            The default is False.
        verbose: bool
            Set to True to obtain more information during runtime.
            The default is False.

        Returns
        -------
        None

        """
        _ = b_i

        # Train PCEs:
        self.pce.fit(X, y, parallel=parallel, verbose=verbose)
        pce_mean, _ = self.pce.eval_metamodel(X)

        # Estimate residuals:
        pce_residuals = {}
        for key, _ in pce_mean.items():
            pce_residuals[key] = pce_mean[key] - y[key]

        self.gpr.fit(X, pce_residuals, parallel=parallel, verbose=verbose)

    @_preprocessing_eval
    def eval_metamodel(self, samples, b_i=0, parallel=False):
        """
        Evaluates metamodel at the requested samples. Calls the eval_metamodel() function
        for PCE and GPRSkl classes

        Parameters
        ----------
        samples: array of shape (n_samples, ndim),
            Samples to evaluate metamodel at.
        b_i: int
            Bootstrapping id

        Returns
        -------
        mean_pred : dict
            Mean of the predictions.
        std_pred : dict
            Standard deviatioon of the predictions.
        """
        _ = b_i

        pce_mean, pce_std = self.pce.eval_metamodel(samples)

        gpr_mean, gpr_std = self.gpr.eval_metamodel(samples)

        mean_pred = {}
        std_pred = {}
        for key in list(pce_mean.keys()):
            mean_pred[key] = pce_mean[key] + gpr_mean[key]
            std_pred[key] = pce_std[key] + gpr_std[key]

        return mean_pred, std_pred
