#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of metamodel as either PC, aPC or GPE
"""

import copy
import os
import warnings

from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from tqdm import tqdm

from .input_space import InputSpace

warnings.filterwarnings("ignore")
# Load the mplstyle
# noinspection SpellCheckingInspection
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


def _preprocessing_fit(fit_function):
    """
    This decorator performs the checks and preprocessing for fitting the
    metamodel.

    Parameters
    ----------
    fit_function : function
        Function with the same signature as self.fit().

    Raises
    ------
    AttributeError
        Inputs X of the fit_function should be a 2D array of the size specified
        by the input space.
    ValueError
        Input y of the fit_function should contain 2D arrays for each key.

    Returns
    -------
    decorator : function
        The decorated function.

    """

    @wraps(fit_function)
    def decorator(self, *args, **kwargs):
        # Use settings
        X = args[0]
        y = args[1]
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]  # args[2]
        if "parallel" in kwargs:
            self.parallel = kwargs["parallel"]  # args[3]
        self.ndim = len(self.input_obj.marginals)

        # Check X
        X = np.array(X)
        n_samples, ndim = X.shape
        if self.ndim != ndim:
            raise AttributeError(
                "The given samples do not match the given number of priors. "
                "The samples should be a 2D array of size"
                "(#samples, #priors)"
            )
        self.n_samples = n_samples

        # Check y
        for key in y.keys():
            y_val = np.array(y[key])
            if y_val.ndim != 2:
                raise ValueError("The given outputs y should be 2D")
            y[key] = np.array(y[key])
        self.out_names = list(y.keys())

        # Build the input space
        if self.input_space is None:
            self.input_space = InputSpace(self.input_obj, self.meta_model_type)
            n_init_samples = X.shape[0]
            self.input_space.n_init_samples = n_init_samples
            self.input_space.init_param_space(np.max(self.pce_deg))

        # Transform input samples
        X = self.input_space.transform(X, method=self.input_transform)

        # Build any other surrogate parts
        self.build_metamodel()

        # --- Loop through data points and fit the surrogate ---
        if self.verbose:
            print(
                f"\n>>>> Training the {self.meta_model_type} metamodel "
                "started. <<<<<<\n"
            )

        # Call the function for fitting
        fit_function(self, X, y, **kwargs)

        if self.verbose:
            print(
                f"\n>>>> Training the {self.meta_model_type} metamodel"
                " sucessfully completed. <<<<<<\n"
            )

    return decorator


def _bootstrap_fit(fit_function):
    """
    Decorator that applies bootstrap iterations around the training of a
    MetaModel.
    Also performs additional transformations to the training outputs, e.g. PCA.

    Parameters
    ----------
    fit_function : function
        Function with the same signature as self.fit().

    Returns
    -------
    decorator : function
        The decorated function.

    """

    @wraps(fit_function)
    def decorator(self, *args, **kwargs):
        X = args[0]
        y = args[1]

        # --- Bootstrap sampling ---
        # Correct number of bootstrap if PCA transformation is required.
        if self.dim_red_method.lower() == "pca" and self.n_bootstrap_itrs == 1:
            self.n_bootstrap_itrs = 100

        # Prepare tqdm iteration maessage
        if self.verbose and self.n_bootstrap_itrs > 1:
            enum_obj = tqdm(
                range(self.n_bootstrap_itrs),
                total=self.n_bootstrap_itrs,
                desc="Bootstrapping the metamodel",
                ascii=True,
            )
        else:
            enum_obj = range(self.n_bootstrap_itrs)

        # Loop over the bootstrap iterations
        for b_i in enum_obj:
            # Do the actual bootstrap
            if b_i > 0:
                self.b_indices = np.random.randint(self.n_samples, size=self.n_samples)
            else:
                self.b_indices = np.arange(len(X))

            # Apply bootstrap to x
            x_train_b = X[self.b_indices]

            # Apply bootstrap to y
            y_train_b = transform_y(self, y, b_i, trafo_type="bootstrap")

            # Apply transformation, e.g. PCA
            # Will be more expensive here than outside this loop,
            # But allows for aspects such as fast bootstrapping
            y_train_b = transform_y(
                self, y_train_b, b_i, trafo_type=self.dim_red_method.lower()
            )

            # Call the fit
            fit_function(self, x_train_b, y_train_b, **kwargs, b_i=b_i)

    return decorator


def transform_y(self, y, b_i=0, trafo_type=""):
    """
    Apply chosen transformation to model outputs per key.
    Currently supports no transform, bootstrap and PCA.

    Parameters
    ----------
    self : object
        An object of class MetaModel that contains needed parameters.
    y : dict
        Output to transform, should contain arrays for each key.
    b_i :  int
        Current bootstrap index. This is used in PCA-transformation to use
        the same number of PCA components in each bootstrap iteration.
        The default is 0.
    trafo_type : string
        The type of transformation to apply. Currently supported are
        'bootstrap': Bootstrap each key with self.b_indices
        'pca': Principal Component Analysis. The transformation has to be
                available in self.
        '': No transformation

    Returns
    -------
    y_transform : dict
        Transformed outputs.

    """
    # Perform bootstrap
    if trafo_type == "bootstrap":
        y_transform = {}
        for key in y:
            y_transform[key] = y[key][self.b_indices]

    # Dimensionality reduction with PCA, if specified
    if trafo_type == "pca":
        y_transform = {}
        for key in y:
            if self.bootstrap_method.lower() == "fast" and b_i > 0:
                self.n_pca_components = self.n_comp_dict[key]

            # Start transformation
            pca, y_transform[key], n_comp = self.pca_transformation(
                y[key], self.n_pca_components
            )
            self.pca[f"b_{b_i + 1}"][key] = pca

            # Store the number of components for fast bootstrapping
            if self.bootstrap_method.lower() == "fast" and b_i == 0:
                self.n_comp_dict[key] = n_comp

    # Do nothing if not specified
    else:
        y_transform = y

    return y_transform


def _preprocessing_eval(eval_function):
    """
    This decorator performs the pre- and postprocessing for evaluating the
    metamodel.

    Parameters
    ----------
    eval_function : function
        Function with the same signature as self.eval_metamodel().

    Returns
    -------
    decorator : function
        The decorated function.

    """

    @wraps(eval_function)
    def decorator(self, *args, **kwargs):
        # Transform into np array - can also be given as list
        samples = args[0]
        samples = np.array(samples)

        # Transform samples to the independent space
        samples = self.input_space.transform(samples, method=self.input_transform)

        out = eval_function(self, samples, **kwargs)

        return out

    return decorator


def _bootstrap_eval(eval_function):
    """
    Decorator that applies bootstrap iterations around the evaluation of a
    MetaModel.
    Also performs additional transformations to the training outputs, e.g. PCA.

    Parameters
    ----------
    eval_function : function
        Function with the same signature as self.eval_metamodel().

    Returns
    -------
    decorator : function
        The decorated function.

    """

    @wraps(eval_function)
    def decorator(self, *args, **kwargs):
        mean_pred_b = {}
        std_pred_b = {}
        # Loop over bootstrap iterations
        for b_i in range(self.n_bootstrap_itrs):
            kwargs["b_i"] = b_i
            mean_pred, std_pred = eval_function(self, *args, **kwargs)

            # Appy inverse transformation
            if self.dim_red_method.lower() == "pca":
                for output, values in mean_pred.items():
                    pca = self.pca[f"b_{b_i + 1}"][output]
                    mean_pred[output] = pca.inverse_transform(values)
                    std_pred[output] = np.zeros(values.shape)

            # Save predictions for each bootstrap iteration
            mean_pred_b[b_i] = mean_pred
            std_pred_b[b_i] = std_pred

        # Change the order of nesting
        mean_pred_all = {}
        for i in sorted(mean_pred_b):
            for k, v in mean_pred_b[i].items():
                if k not in mean_pred_all:
                    mean_pred_all[k] = [None] * len(mean_pred_b)
                mean_pred_all[k][i] = v

        # Compute the moments of predictions over the predictions
        for output in self.out_names:
            # Only use bootstraps with finite values
            finite_rows = np.isfinite(mean_pred_all[output]).all(axis=2).all(axis=1)
            outs = np.asarray(mean_pred_all[output])[finite_rows]

            # Compute mean and stdev
            mean_pred[output] = np.mean(outs, axis=0)
            if self.n_bootstrap_itrs > 1:
                std_pred[output] = np.std(outs, axis=0)
            else:
                std_pred[output] = std_pred_b[self.n_bootstrap_itrs - 1][output]

        return mean_pred, std_pred

    return decorator


class MetaModel:
    """
    Meta (surrogate) model base class

    This class describes the necessary functions and propoerties of a
    surrogate model in bayesvalidrox. It accepts an input object (input_obj)
    containing the specification of the distributions for uncertain parameters.

    Attributes
    ----------
    input_obj : obj
        Input object with the information on the model input parameters.
    bootstrap_method : str
        Bootstraping method. Options are `'normal'` and `'fast'`. The default
        is `'fast'`. It means that in each iteration except the first one, only
        the coefficent are recalculated with the ordinary least square method.
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
    is_gaussian : bool
        Set to True if the surrogate model returns mean and stdev.
        The default is `False`.
    verbose : bool
        Prints summary of the regression results. Default is `False`.
    n_mc : int
        Number of Monte Carlo samples used for the calculation of the moments. Standard is 1000.

    """

    def __init__(
        self,
        input_obj,
        meta_model_type="",
        bootstrap_method="fast",
        n_bootstrap_itrs=1,
        dim_red_method="no",
        is_gaussian=False,
        verbose=False,
        n_mc=1000,
        input_transform="user",
    ):

        # Inputs
        self.input_obj = input_obj
        self.meta_model_type = meta_model_type
        self.bootstrap_method = bootstrap_method
        self.n_bootstrap_itrs = n_bootstrap_itrs
        self.dim_red_method = dim_red_method
        self.is_gaussian = is_gaussian
        self.verbose = verbose
        self.n_mc = n_mc
        self.input_transform = input_transform

        # Other params
        self.input_space = None
        self.par_names = []
        self.out_names = []
        self.n_samples = None
        self.lc_error = None
        self.loocv_score_dict = None
        self.ndim = len(self.input_obj.marginals)

        # Parameters specific to bootstrap
        self.n_comp_dict = {}
        self.first_out = {}

        # Parameters specific to PCA
        self.pca = self.AutoVivification()
        self.var_pca_threshold = None
        self.n_pca_components = None

        # Parameters for Inference
        self.rmse = None

        # Build general parameters
        self.pce_deg = 1

        # General warnings
        if not self.is_gaussian:
            print(
                "There are no estimations of surrogate uncertainty available"
                " for the chosen regression options. This might lead to issues"
                " in later steps."
            )

    def check_is_gaussian(self) -> bool:
        """
        Check if the current surrogate will return both a mean and stdev as
        the output of being evaluated.

        This function should be extended and applied in the constructor of
        all child classes!

        Returns
        -------
        bool
            True if stdev is also returned.
        """
        return None

    def add_parnames(self):
        """
        Collect all names of input parameters in property self.par_names.
        """
        if len(self.par_names) == 0:
            for m in self.input_obj.marginals:
                self.par_names.append(m.name)

    def build_metamodel(self) -> None:
        """
        Builds the parts for the metamodel (polynomes,...) that are needed
        before fitting.

        This function should be extended and called in/before training the
        surrogate child classes!

        Returns
        -------
        None

        """
        return None

    @_preprocessing_fit
    def fit(self, X: np.array, y: dict, parallel=False, verbose=False):
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
        _ = (X, y, parallel, verbose)

    # -------------------------------------------------------------------------

    def pca_transformation(self, target, n_pca_components):
        """
        Transforms the targets (outputs) via Principal Component Analysis.
        The number of features is set by `self.n_pca_components`.
        If this is not given, `self.var_pca_threshold` is used as a threshold.

        ToDo: Check the inputs needed for this class, there is an error when PCA is used.
        ToDo: From the y_transformation() function, a dictionary is being sent
        instead of an array for target.

        Parameters
        ----------
        target : array of shape (n_samples,)
            Target values.

        Returns
        -------
        pca : obj
            Fitted sklearnPCA object.
        OutputMatrix : array of shape (n_samples,)
            Transformed target values.
        n_pca_components : int
            Number of selected principal components.

        """
        # Get current shape of the outputs
        n_samples, n_features = target.shape

        # Switch to var_pca if n_pca_components is too large
        if (n_pca_components is not None) and (n_pca_components > n_features):
            n_pca_components = None
            if self.verbose:
                print("")
                print(
                    "WARNING: Too many components are set for PCA. The transformation "
                    "will proceed based on explainable variance."
                )

        # Calculate n_pca_components based on decomposition of the variance
        if n_pca_components is None:
            if self.var_pca_threshold is not None:
                var_pca_threshold = self.var_pca_threshold
            else:
                var_pca_threshold = 99  # 100.0
            # Instantiate and fit sklearnPCA object
            covar_matrix = sklearnPCA(n_components=None)
            covar_matrix.fit(target)
            var = np.cumsum(
                np.round(covar_matrix.explained_variance_ratio_, decimals=5) * 100
            )
            # Find the number of components to explain self.varPCAThreshold of
            # variance
            try:
                n_components = np.where(var >= var_pca_threshold)[0][0] + 1
            except IndexError:
                n_components = min(n_samples, n_features)

            n_pca_components = min(n_samples, n_features, n_components)

        # Set the solver to 'auto' if no reduction is wanted
        # Otherwise use 'arpack'
        solver = "auto"
        if n_pca_components < n_features:
            solver = "arpack"

        # Fit and transform with the selected number of components
        pca = sklearnPCA(n_components=n_pca_components, svd_solver=solver)
        scaled_target = pca.fit_transform(target)

        return pca, scaled_target, n_pca_components

    # -------------------------------------------------------------------------

    def eval_metamodel(self, samples):
        """
        Evaluates metamodel at the requested samples. One can also generate
        nsamples.

        Parameters
        ----------
        samples : array of shape (n_samples, ndim), optional
            Samples to evaluate metamodel at. The default is None.

        Returns
        -------
        mean_pred : dict
            Mean of the predictions.
        std_pred : dict
            Standard deviatioon of the predictions. Return None if
            self.is_gaussian == False
        """
        _ = samples
        return {}, {}

    def calculate_moments(self):
        """
        Computes the first two moments of the metamodel.

        Returns
        -------
        means: dict
            The first moment (mean) of the surrogate.
        stds: dict
            The second moment (standard deviation) of the surrogate.

        """
        # Compute the random Monte Carlo Samples with the random sampler method
        samples = self.input_space.random_sampler(self.n_mc)

        # Calculate the mean and the standard deviation of the samples by using eval_meatmodel
        mean_pred, _ = self.eval_metamodel(samples)
        print(f"mean_pred: {type(mean_pred)}")

        # Calculate the moments by calculating the numpy mean of the mean over all variables
        rmse_mean = {}
        rmse_std = {}

        for key, mean in mean_pred.items():
            rmse_mean[key] = np.mean(mean, axis=0)
            rmse_std[key] = np.std(mean, axis=0)

        print(f"moments_mean: {rmse_mean}")
        print(f"moments_std: {rmse_std}")

        return rmse_mean, rmse_std

    # -------------------------------------------------------------------------

    class AutoVivification(dict):
        """
        Implementation of perl's AutoVivification feature.

        Source: https://stackoverflow.com/a/651879/18082457
        """

        def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    # -------------------------------------------------------------------------

    def copy_meta_model_opts(self):
        """
        This method is a convinient function to copy the metamodel options.

        Returns
        -------
        metamod_copy : object
            The copied object.

        """
        metamod_copy = copy.deepcopy(self)
        metamod_copy.input_obj = self.input_obj
        metamod_copy.input_space = self.input_space
        metamod_copy.ndim = len(self.input_obj.marginals)
        return metamod_copy

    # -------------------------------------------------------------------------

    def add_input_space(self):
        """
        Instanciates experimental design object.

        Returns
        -------
        None.

        """
        self.input_space = InputSpace(self.input_obj)
