#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of metamodel as GPE, using the Scikit-Learn library
"""

import math
import os
import warnings

import functools
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from .meta_model import (
    MetaModel,
    _preprocessing_fit,
    _bootstrap_fit,
    _preprocessing_eval,
    _bootstrap_eval,
)

warnings.filterwarnings("ignore")
# warnings.simplefilter('default')
# Load the mplstyle
# noinspection SpellCheckingInspection
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


class MySklGPE(GaussianProcessRegressor):
    """
    GP ScikitLearn class, to change the default values for maximum iterations
    and optimization tolerance.
    """

    def __init__(self, max_iter=10_000, gtol=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.gtol = gtol


class GPESkl(MetaModel):
    """
    GP MetaModel using the Scikit-Learn library

    This class trains a surrogate model of type Gaussian Process Regression.
    It accepts an input object (input_obj)
    containing the specification of the distributions for uncertain parameters
    and a model object with instructions on how to run the computational model.

    Attributes
    ----------
    input_obj : obj
        Input object with the information on the model input parameters.
    meta_model_type : str
        Surrogate model types, in this case GPE.
        Default is GPE.
    gpe_reg_method : str
        GPE regression method to compute the kernel hyperparameters. The following
        regression methods are available for Scikit-Learn library
        1. LBFGS:
        Default is `LBFGS`.
    auto_select: bool
        Flag to loop through different available Kernels and select the best one
        based on BME criteria.
        Default is False.
    kernel_type: str
        Type of kernel to use and train for. The following Scikit-Learn kernels are available:
        1. RBF: Squared exponential kernel
        2. Matern: Matern kernel
        3. RQ: rational quadratic kernel
        Default is `'RBF'` kernel.
    n_restarts: int
        Number of multiple starts to do for each GP training.
        Default is 10
    isotropy: bool
        Flag to train an isotropic kernel (one length scale for all input parameters) or
         an anisotropic kernel (a length scale for each dimension). True for isotropy,
         False for anisotropic kernel
        Default is True
    noisy: bool
        True to consider a WhiteKernel for regularization purposes, and optimize for
        the noise hyperparameter.
        Default is False
    nugget: float
        Constant value added to the Kernel matrix for regularization purposes (not optimized)
        Default is 1e-9
    normalize_x_method: str
        Type of transformation to apply to the inputs. If None or `none`,
        no transformation is done.
        The followint options are available:
        1. 'norm': normalizes inputs U[0,1]
        2. 'standard': standarizes inputs N[0, 1]
        3. 'none': no transformation is done
        3. None: No transformation is done.
    n_bootstrap_itrs : int
        Number of iterations for the bootstrap sampling. The default is `1`.
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
    input_transform: str
        Type of transformation to apply to the inputs. Default is user.
    """

    def __init__(
        self,
        input_obj,
        meta_model_type="GPE",
        gpe_reg_method="LBFGS",
        n_restarts=10,
        auto_select=False,
        kernel_type="RBF",
        isotropy=True,
        noisy=False,
        nugget=1e-9,
        n_bootstrap_itrs=1,
        normalize_x_method="norm",
        transform_y_method='norm',
        dim_red_method="no",
        verbose=False,
        input_transform="user",
    ):

        # Check if the surrogate outputs gaussian results: Always TRUE for GPs
        is_gaussian = self.check_is_gaussian()

        # Use parent init
        super().__init__(
            input_obj=input_obj,
            meta_model_type=meta_model_type,
            n_bootstrap_itrs=n_bootstrap_itrs,
            dim_red_method=dim_red_method,
            is_gaussian=is_gaussian,
            verbose=verbose,
            input_transform=input_transform,
        )

        # Additional inputs
        # Parameters that are not needed from the outside are denoted with '_'
        self.meta_model_type = meta_model_type
        self._gpe_reg_method = gpe_reg_method
        self.regression_dict = {}
        self._gpe_reg_options = {}

        self._auto_select = auto_select
        self._kernel_isotropy = isotropy
        self._kernel_noise = noisy
        self._kernel_type = kernel_type
        self._nugget = nugget
        self.n_restarts = n_restarts
        self.normalize_x_method = normalize_x_method
        self.transform_y_method = transform_y_method

        # Other params
        self._gp_poly = None
        self._x_scaler = None
        self._bme_score = None
        self._kernel_name_dict = None

    def check_is_gaussian(self) -> bool:
        """
        Check if the metamodel returns a mean and stdev.

        Returns
        -------
        bool
            TRUE

        """
        return True

    def build_metamodel(self) -> None:
        """
        Builds the parts for the metamodel (,...) that are needed before fitting.
        This is executed outside any loops related to e.g. bootstrap or
        transformations such as pca.

        Returns
        -------
        None

        """
        # Initialize the nested dictionaries
        self._gp_poly = self.AutoVivification()
        self._x_scaler = self.AutoVivification()
        self._bme_score = self.AutoVivification()
        self._kernel_name_dict = self.AutoVivification()
        # self._lc_error = self.AutoVivification()

    def build_kernels(self):
        """
        Initializes the different possible kernels, and selects the ones to train for,
        depending on the input options.
        It needs the self._auto_select variable, which must be a boolean,
        the self.kernel_type variable (which must a string with one of the [currently]
        3 valid Kernel types.) If an
        invalid kernel type is given, the RBF kernel is used).
        and the self.isotropy variable. If True, it initializes isotropic kernels.

        Raises
        ------
        AttributeError: if an invalid type of Kernel is given
        TypeError: if the kernel type is not a string

        Returns
        -------
        List: with the kernels to iterate over
        List: with the names of the kernels to iterate over
        """
        _ndim = self.input_space.ndim

        # Set boundaries for length scales:
        value = np.empty((), dtype=object)
        value[()] = [1e-3, 1e3]  # [1e-5, 1e5] CHANGED

        if self._kernel_isotropy:
            # Isotropic Kernel
            ls_bounds = list(np.full(1, value, dtype=object))
            ls_values = 1
        else:
            ls_bounds = list(np.full(_ndim, value, dtype=object))
            ls_values = list(np.full(_ndim, 1))

        # Generate list of probable kernels:
        rbf_kernel = 1 * kernels.RBF(
            length_scale=ls_values, length_scale_bounds=ls_bounds
        )
        matern_kernel = 1 * kernels.Matern(
            length_scale=ls_values, length_scale_bounds=ls_bounds, nu=1.5
        )
        rq_kernel = 1 * kernels.RationalQuadratic(
            length_scale=ls_values, length_scale_bounds=ls_bounds, alpha=1
        )
        kernel_dict = {"RBF": rbf_kernel, "Matern": matern_kernel, "RQ": rq_kernel}

        if self._auto_select:
            kernel_list = list(kernel_dict.values())
            kernel_names = list(kernel_dict.keys())
        else:
            try:
                kernel_list = [kernel_dict[self._kernel_type]]
                kernel_names = [self._kernel_type]
            except Exception as exc:
                if isinstance(self._kernel_type, str):
                    raise AttributeError(
                        f"The kernel option {self._kernel_type} is not available."
                    ) from exc
                raise TypeError(
                    f"The kernel option {self._kernel_type} is of an invalid type."
                ) from exc

        return kernel_list, kernel_names

    def transform_x(self, X: np.array, transform_type=None):
        """
        Scales the inputs (X) during training using either normalize ([0, 1]),
        or standardize (N[0, 1]).
        If None, then the inputs are not scaled. If an invalid transform_type
        is given, no transformation is done.
        Parameters
        ----------
        X: 2D list or np.array of shape (#samples, #dim)
        The parameter value combinations to train the model with.
        transform_type: str
            Transformation to apply to the input parameters. Default is None
        Raises
        ------
        AttributeError: If an invalid scaling name is given.
        Returns
        -------
        np.array: (#samples, #dim)
            transformed input parameters
        obj: Scaler object
            transformation object, for future transformations during surrogate evaluation

        """
        if transform_type is None:
            transform_type = self.normalize_x_method

        if transform_type is None or transform_type.lower() == "none":
            scaler = None
            return X, scaler
        if transform_type.lower() == "norm":
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(X)
        elif transform_type.lower() == "standard":
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(X)
        else:
            raise AttributeError(f"No scaler {transform_type} found.")
        return x_scaled, scaler

    @_preprocessing_fit
    @_bootstrap_fit
    def fit(self, X: np.array, y: dict, parallel=False, verbose=False, b_i=0):
        """
        Fits the surrogate to the given data (samples X, outputs y).
        Note here that the samples X should be the transformed samples provided
        by the experimental design if the transformation is used there.

        Parameters
        ----------
        X : 2D list or np.array of shape (#samples, #dim)
            The parameter value combinations that the model was evaluated at.
        y : dict of 2D lists or arrays of shape (#samples, #timesteps)
            The respective model evaluations.
        parallel : bool
            Set to True to run the training in parallel for various keys.
            The default is False.
        verbose : bool
            Set to True to obtain more information during runtime.
            The default is False.

        Returns
        -------
        None.

        """

        # For loop over the components/outputs
        if self.verbose and self.n_bootstrap_itrs == 1:
            items = tqdm(y.items(), desc="Fitting regression")
        else:
            items = y.items()

        # Transform inputs:
        x_scaled, scaler = self.transform_x(X=X)
        self._x_scaler[f"b_{b_i + 1}"] = scaler

        for key, output in items:
            # Parallel fit regression
            out = None
            # CHANGED: Changed parallel option 
            if parallel:  # and (not self.fast_bootstrap or b_i == 0):
                out = Parallel(n_jobs=6, backend="loky", batch_size="auto")(
                    delayed(self.adaptive_regression)(
                        x_scaled, output[:, idx], idx, self.verbose
                    )
                    for idx in range(output.shape[1])
                )
            elif not parallel:  # and (not self.fast_bootstrap or b_i == 0):
                results = map(
                    functools.partial(
                        self.adaptive_regression, x_scaled, verbose=self.verbose
                    ),
                    [output[:, idx] for idx in range(output.shape[1])],
                    range(output.shape[1]),
                )
                out = list(results)

            # Create a dict to pass the variables
            for i in range(output.shape[1]):
                self._gp_poly[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["gp"]
                self._bme_score[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["bme"]
                self._kernel_name_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i][
                    "kernel_name"
                ]

    # ----------------------------------------------------------------------------
    def adaptive_regression(self, X, y, var_idx, verbose=False):
        """
        Adaptively fits the GPE model by comparing different Kernel options

        Parameters
        ----------
        X : array of shape (n_samples, ndim)
            Training set. These samples should be already transformed.
        y : array of shape (n_samples,)
            Target values, i.e. simulation results for the Experimental design.
        var_idx : int
            Index of the output.
        verbose : bool, optional
            Print out summary. The default is False.

        Returns
        -------
        return_vars : Dict
            Fitted estimator, BME score

        """
        # CHANGED: Added transform_y_method
    
        if self.transform_y_method is None or self.transform_y_method.lower() == "none":
            y_norm=False
        elif self.transform_y_method.lower() == 'norm':
            y_norm = True
        else:
            raise AttributeError(f"No transformation method {self.transform_y_method} found for the outputs.")

        # Initialization
        gp_list = {}
        bme = []

        # Get kernels:
        kernel_list, kernel_names = self.build_kernels()

        for i, kernel in enumerate(kernel_list):
            if self._kernel_noise:
                kernel = kernel + kernels.WhiteKernel(
                    noise_level=np.std(y) / math.sqrt(2)
                )

            # Set n_restars as variable?
            # gp_list[i] = MySklGPE(kernel=kernel,
            #                       n_restarts_optimizer=20,
            #                       alpha=self._nugget,
            #                       normalize_y=True)
            #
            gp_list[i] = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                alpha=self._nugget,
                normalize_y=y_norm,
            )

            # Fit to data using Maximum Likelihood Estimation
            gp_list[i].fit(X, y)

            # Store the MLE as BME score
            bme.append(gp_list[i].log_marginal_likelihood())

        # Select the GP with the highest bme
        idx_max = np.argmax(bme)
        gp = gp_list[idx_max]

        if var_idx is not None and verbose:
            gp_score = gp.score(X, y)
            print("=" * 50)
            print(" " * 10 + " Summary of results ")
            print("=" * 50)

            print(f"Output variable {var_idx}:")
            print("The estimation of GPE coefficients converged,")
            print(f"with the R^2 score: {gp_score:.3f} ")
            print(f"using a {kernel_names[idx_max]} Kernel")
            print("=" * 50)

        # Create a dict to pass the outputs
        return_vars = {}
        return_vars["gp"] = gp
        return_vars["bme"] = bme[idx_max]
        return_vars["kernel_name"] = kernel_names[idx_max]

        return return_vars

    # -------------------------------------------------------------------------
    @staticmethod
    def scale_x(X: np.array, transform_obj: object):
        """
        Transforms the inputs based on the scaling done during training
        Parameters
        ----------
        X: 2D list or np.array of shape (#samples, #dim)
            The parameter value combinations to evaluate the model with.
        transform_obj: Scikit-Learn object
            Class instance to transform inputs

        Returns
        -------
        np.array (#samples, #dim)
            Transformed input sets
        """
        if transform_obj is None:
            x_t = X
        else:
            x_t = transform_obj.transform(X)
        return x_t

    @_preprocessing_eval
    @_bootstrap_eval
    def eval_metamodel(self, samples, b_i=0, parallel=False):
        """
        Evaluates GP metamodel at the requested samples.
        CHANGED: Added parallel option to run the evaluation in parallel for various keys.
        Parameters
        ----------
        samples : array of shape (n_samples, ndim), optional
            Samples to evaluate metamodel at. The default is None.
        b_i : int, optional
            Index of the bootstrap iteration to evaluate. The default is 0.
        parallel : bool, optional
            Set to True to run the evaluation in parallel for various keys.
            The default is False.
        Returns
        -------
        mean_pred : dict
            Mean of the predictions.
        std_pred : dict
            Standard deviation of the predictions.
        """
        # Transform the input parameters
        samples_sc = self.scale_x(
            X=samples, transform_obj=self._x_scaler[f"b_{b_i + 1}"]
        )

        # Extract model dictionary
        model_dict = self._gp_poly[f"b_{b_i + 1}"]

        # Loop over outputs
        mean_pred = {}
        std_pred = {}

        for output, values in model_dict.items():
            mean = np.empty((len(samples), len(values)))
            std = np.empty((len(samples), len(values)))

            if parallel:
                # 1. Gather all GP models for this specific output
                gps_to_eval = [
                    self._gp_poly[f"b_{b_i + 1}"][output][in_key] 
                    for in_key in values.keys()
                ]
                
                # 2. Run predictions in parallel
                # We delay the built-in .predict() method of the scikit-learn GP
                results = Parallel(n_jobs=6, backend="loky", batch_size="auto")(
                    delayed(gp.predict)(samples_sc, return_std=True) 
                    for gp in gps_to_eval
                )
                
                # 3. Unpack the parallel results into the pre-allocated arrays
                for idx, (y_mean, y_std) in enumerate(results):
                    mean[:, idx] = y_mean
                    std[:, idx] = y_std
                    
            else:
                # Fallback to sequential evaluation
                idx = 0
                for in_key, _ in values.items():
                    gp = self._gp_poly[f"b_{b_i + 1}"][output][in_key]
                    y_mean, y_std = gp.predict(samples_sc, return_std=True)
                    mean[:, idx] = y_mean
                    std[:, idx] = y_std
                    idx += 1

            # idx = 0
            # for in_key, _ in values.items():

            #     gp = self._gp_poly[f"b_{b_i + 1}"][output][in_key]

            #     # Prediction
            #     y_mean, y_std = gp.predict(samples_sc, return_std=True)

            #     mean[:, idx] = y_mean
            #     std[:, idx] = y_std
            #     idx += 1

            mean_pred[output] = mean
            std_pred[output] = std

        return mean_pred, std_pred
