#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of metamodel as PC or aPC
"""

import copy
import os
import warnings
import math
import json

import functools
from functools import wraps
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from joblib import Parallel, delayed
from tqdm import tqdm

from .apoly_construction import apoly_construction
from .bayes_linear import VBLinearRegression, EBLinearRegression
from .eval_rec_rule import eval_univ_basis
from .glexindex import glexindex
from .orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from .reg_fast_ard import RegressionFastARD
# NEW: Import extended FastARD
from .reg_fast_ard_extended import RegressionFastARDExtended
from .reg_fast_laplace import RegressionFastLaplace

from .meta_model import (
    MetaModel,
    _preprocessing_fit,
    _bootstrap_fit,
    _preprocessing_eval,
    _bootstrap_eval,
)

from .supplementary import corr_loocv_error, create_psi

warnings.filterwarnings("ignore")
# Load the mplstyle
# noinspection SpellCheckingInspection
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


def calc_param_combinations(par_names):
    """
    Generate all possible unique combinations of
    the input parameters of the given engine, for
    all possible group sizes.

    Returns
    -------
    sobol_par : dict
        Lists of combinations for any possible
        parameter group size.

    """
    par_cmb = {}
    par = par_names
    for i in range(len(par)):
        par_cmb[i + 1] = [list(comb) for comb in combinations(par, i + 1)]
    return par_cmb


class PINV:
    """
    (Pseudo)-inverse class for training PCE.

    """

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        """
        Calculate coefficients c in X*c=y.

        Parameters
        ----------
        X : np.ndarray
            Matrix on the left hand side.
        y : np.ndarray
            Vector on the right hand side.

        """
        p_inv = np.linalg.pinv(X)
        self.coef_ = np.matmul(p_inv, y)

    def predict(self, psi):
        """
        Calculates Psi*c.

        """
        return np.dot(psi, self.coef_)


def decorator_deriv_check(derivative):
    """
    Performs the preprocessing, input checks and general calculations
    for all function that calculate derivatives.

    Parameters
    ----------
    derivative : function

    Raises
    ------
    AttributeError
        - Derivatives do not currently support GPE
        - Derivatives not compatible with PCA at the moment!
        - Derivatives only useable for arbitrary-type polynomials!
        - The evaluation points should be a list or np.array!
    ValueError
        - The parameter {par_name} chosen for the derivative is not valid!

    Returns
    -------
    decorator
        The decorator.

    """

    @wraps(derivative)
    def decorator(self, *args, **kwargs):
        samples = args[0]
        deriv_names = args[1]
        deriv_string = json.dumps(deriv_names)

        if self.meta_model_type.lower() == "gpe":
            return AttributeError("Derivatives do not currently support GPE!")
        if self.dim_red_method.lower() == "pca":
            raise AttributeError("Derivatives not compatible with PCA at the moment!")
        if not (self.input_space.input_data_given or not self.input_space.pce):
            raise AttributeError(
                "Derivatives only useable for arbitrary-type polynomials!"
            )

        # Check the inputs and transform to independent space
        try:
            samples = np.array(samples)
        except AttributeError:
            print("The evaluation points should be a list or np.array!")
        if samples.ndim != 2:
            samples = samples.reshape(1, len(samples))

        for par_name in deriv_names:
            if par_name not in self.par_names:
                raise ValueError(
                    f"The parameter {par_name} chosen for the derivative is not valid!"
                )

        # Setup dict for derivative degrees (for non-derivative univariate polynomials)
        if deriv_string not in list(self.deriv_deg.keys()):
            self.deriv_deg[deriv_string] = None

        # Setup dict for derivative coefficients
        if deriv_string not in list(self.deriv_coeff.keys()):
            self.deriv_coeff[deriv_string] = None
        self.build_outer_deriv(deriv_names)

        # Compute degrees for derivative univariate polynomials
        for par_name in deriv_names:
            degree = deriv_names[par_name]
            if not degree in list(self.deriv_inner_deg.keys()):
                self.deriv_inner_deg[degree] = {}
            if not par_name in list(self.deriv_inner_deg[degree].keys()):
                self.deriv_inner_deg[degree][par_name] = None
        self.build_deriv_inner_deg(deriv_names)

        # Compute derivatives of all univariate bases recursively
        if 0 not in self.der_basis_coeffs:
            self.der_basis_coeffs[0] = copy.deepcopy(self._polycoeffs)
        max_deg = 0
        for key in deriv_names:
            max_deg = max(max_deg, deriv_names[key])
        if max_deg not in self.der_basis_coeffs:
            self.der_basis_coeffs = self.build_der_basis_coeffs(
                max_deg, self.der_basis_coeffs
            )

        out = derivative(self, *args, **kwargs)
        return out

    return decorator


class PCE(MetaModel):
    """
    PCE MetaModel

    This class trains a surrogate model of type Polynomial Chaos Expansion.
    It accepts an input object (input_obj)
    containing the specification of the distributions for uncertain parameters
    and a model object with instructions on how to run the computational model.

    Attributes
    ----------
    input_obj : obj
        Input object with the information on the model input parameters.
    meta_model_type : str
        PCE-surrogate model types. Two surrogate model types are supported:
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
        # NEW: Added FastARDExtended method mimicking Matlab
        6. FastARDExtended: Extended Fast Bayesian ARD Regression 
        7. VBL: Variational Bayesian Learning
        8. EBL: Emperical Bayesian Learning
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
        meta_model_type="PCE",
        pce_reg_method="OLS",
        bootstrap_method="fast",
        n_bootstrap_itrs=1,
        pce_deg=1,
        pce_q_norm=1.0,
        dim_red_method="no",
        verbose=False,
        input_transform="user",
    ):

        # Check if the surrogate outputs gaussian results
        self.pce_reg_method = pce_reg_method
        self.n_bootstrap_itrs = n_bootstrap_itrs
        is_gaussian = self.check_is_gaussian()

        # Use parent init
        super().__init__(
            input_obj,
            meta_model_type,
            bootstrap_method,
            n_bootstrap_itrs,
            dim_red_method,
            is_gaussian,
            verbose,
            input_transform,
        )

        # Additional inputs
        # Parameters that are not needed from the outside are denoted with '_'
        self.pce_deg = pce_deg
        self.pce_q_norm = pce_q_norm
        self.regression_dict = {}
        self._pce_reg_options = {}

        # Other params
        self._polycoeffs = None
        self._univ_p_val = None
        self._all_basis_indices = None
        self._deg_array = None
        self.lc_error = None
        self._clf_poly = None
        self._basis_dict = None
        self._coeffs_dict = None
        self._q_norm_dict = None
        self._deg_dict = None

        self.sobol = None
        self.total_sobol = None

        self.deriv_inner_deg = {}
        self.deriv_coeff = {}
        self.deriv_deg = {}
        self.der_basis_coeffs = {}

        # Initialize the regression options as a dictionary
        self.set_regression_options()

    def check_is_gaussian(self) -> bool:
        """
        Check if the metamodel returns a mean and stdev.

        Returns
        -------
        bool
            True if the metamodel can return estimations of uncertainty
            with each prediction. False otherwise.

        """
        if self.n_bootstrap_itrs > 1:
            return True
        if self.pce_reg_method.lower() in [
            "ols",
            "lars",
            "lassolars",
            "lassolarscv",
            "pinv",
            "omp",
        ]:
            return False
        return True

    def set_regression_options(self) -> None:
        """
        Collects the generic settings for the regression in a dictionary.
        This includes the regression objects and their arguments.

        Returns
        -------
        None

        """
        # Collect all regression objects
        self.regression_dict["ols"] = lm.LinearRegression
        self.regression_dict["pinv"] = PINV
        self.regression_dict["brr"] = lm.BayesianRidge
        self.regression_dict["ard"] = lm.ARDRegression
        self.regression_dict["fastard"] = RegressionFastARD
        # NEW: Register FastARDExtended
        self.regression_dict["fastardextended"] = RegressionFastARDExtended
        self.regression_dict["bcs"] = RegressionFastLaplace
        self.regression_dict["lassolarscv"] = lm.LassoLarsCV
        self.regression_dict["lassolars"] = lm.LassoLars
        self.regression_dict["lars"] = lm.Lars
        self.regression_dict["sgdr"] = lm.SGDRegressor
        self.regression_dict["omp"] = OrthogonalMatchingPursuit
        self.regression_dict["vbl"] = VBLinearRegression
        self.regression_dict["ebl"] = EBLinearRegression

        # Collect the corresponding options
        self._pce_reg_options["ols"] = {"fit_intercept": False}
        self._pce_reg_options["pinv"] = {}
        self._pce_reg_options["brr"] = {
            #"n_iter": 1000,
            "tol": 1e-7,
            "fit_intercept": False,
            # 'compute_score':compute_score,
            "alpha_1": 1e-04,
            "alpha_2": 1e-04,
            # 'lambda_1':lambda_, 'lambda_2':lambda_
        }
        self._pce_reg_options["ard"] = {
            "fit_intercept": False,
            # 'compute_score':compute_score,
            #"n_iter": 1000,
            "tol": 0.0001,
            "alpha_1": 1e-3,
            "alpha_2": 1e-3,
            # 'lambda_1':lambda_, 'lambda_2':lambda_
        }
        self._pce_reg_options["fastard"] = {
            "fit_intercept": False,
            "normalize": True,
            # 'compute_score':compute_score,
            "n_iter": 3000,
            "tol": 1e-3,
        }
        # NEW: Options for FastARDExtended
        self._pce_reg_options["fastardextended"] = {
            "fit_intercept": False,
            "normalize": True,
            "n_iter": 3000,
            "tol": 1e-3,
        }
        self._pce_reg_options["bcs"] = {
            "fit_intercept": False,
            # 'bias_term':bias_term,
            "n_iter": 1000,
            "tol": 1e-7,
        }
        self._pce_reg_options["lars"] = {"fit_intercept": False}
        self._pce_reg_options["lassolars"] = {"fit_intercept": False}
        self._pce_reg_options["lassolarscv"] = {"fit_intercept": False}
        self._pce_reg_options["sgdr"] = {
            "fit_intercept": False,
            "max_iter": 1000,
            "tol": 1e-7,
        }
        self._pce_reg_options["omp"] = {"fit_intercept": False}
        self._pce_reg_options["vbl"] = {"fit_intercept": False}
        self._pce_reg_options["ebl"] = {"optimizer": "em"}
    
    def add_parnames(self):
        """
        Collect all names of input parameters in property self.par_names.
        """
        if len(self.par_names) == 0:
            for m in self.input_obj.marginals:
                self.par_names.append(m.name)
   
    def build_metamodel(self) -> None:
        """
        Builds the parts for the metamodel (polynomes,...) that are needed before fitting.
        This is executed outside of any loops related to e.g. bootstrap or
        transformations such as pca.

        Returns
        -------
        None

        """
        # Generate polynomials
        self._generate_polynomials(np.max(self.pce_deg))
        self.add_parnames()

        # Initialize the nested dictionaries
        self._deg_dict = self.AutoVivification()
        self._q_norm_dict = self.AutoVivification()
        self._coeffs_dict = self.AutoVivification()
        self._basis_dict = self.AutoVivification()
        self.loocv_score_dict = self.AutoVivification()
        self._clf_poly = self.AutoVivification()
        self.lc_error = self.AutoVivification()

        self._deg_array = self._select_degree()  # self.ndim, self.n_samples)

        # Generate all basis indices
        self._all_basis_indices = self.AutoVivification()
        for deg in self._deg_array:
            keys = self._all_basis_indices.keys()
            if deg not in np.fromiter(keys, dtype=float):
                # Generate the polynomial basis indices
                for _, q in enumerate(self.pce_q_norm):
                    basis_indices = glexindex(
                        start=0,
                        stop=deg + 1,
                        dimensions=self.ndim,
                        cross_truncation=q,
                        reverse=False,
                        graded=True,
                    )
                    self._all_basis_indices[str(deg)][str(q)] = basis_indices

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
        for key, output in items:

            # Parallel fit regression
            out = None
            if parallel and (self.bootstrap_method != "fast" or b_i == 0):
                # CHANGED: Changed parallel option
                out = Parallel(n_jobs=6, backend="loky", batch_size="auto")(
                    delayed(self.adaptive_regression)(X, output[:, idx], idx)
                    for idx in range(output.shape[1])
                )
            elif not parallel and (self.bootstrap_method != "fast" or b_i == 0):
                results = map(
                    functools.partial(self.adaptive_regression, X),
                    [output[:, idx] for idx in range(output.shape[1])],
                    range(output.shape[1]),
                )
                out = list(results)

            # Store the first out dictionary
            if self.bootstrap_method == "fast" and b_i == 0:
                self.first_out[key] = copy.deepcopy(out)

            # Update the coefficients with OLS during bootstrap-iters
            if b_i > 0 and self.bootstrap_method == "fast":
                out = self.update_pce_coeffs(X, output, self.first_out[key])

            # Create a dict to pass the variables
            for i in range(output.shape[1]):
                self._deg_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["degree"]
                self._q_norm_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["qnorm"]
                self._coeffs_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["coeffs"]
                self._basis_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i][
                    "multi_indices"
                ]
                self.loocv_score_dict[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i][
                    "loo_cv_score"
                ]
                self._clf_poly[f"b_{b_i + 1}"][key][f"y_{i + 1}"] = out[i]["clf_poly"]
                self.lc_error[f"b_{b_i+1}"][key][f"y_{i+1}"] = out[i]["lc_error"]

    # -------------------------------------------------------------------------

    def update_pce_coeffs(self, X, y, out_dict=None):
        """
        Updates the PCE coefficents using only the ordinary least square method
        for the fast version of the bootstrapping.

        Parameters
        ----------
        X : array of shape (n_samples, ndim)
            Training set. These samples should be already transformed.
        y : array of shape (n_samples, n_outs)
            The (transformed) model responses.
        out_dict : dict
            The training output dictionary of the first iteration, i.e.
            the surrogate model for the original experimental design.

        Returns
        -------
        final_out_dict : dict
            The updated training output dictionary.

        """
        # Make a copy
        final_out_dict = copy.deepcopy(out_dict)

        # Create the _univ_p_val based on the given X
        _univ_p_val = self.univ_basis_vals(X, n_max=np.max(self.pce_deg))

        # Loop over the points
        for i in range(y.shape[1]):

            # Extract nonzero basis indices
            nnz_idx = np.nonzero(out_dict[i]["coeffs"])[0]
            if len(nnz_idx) != 0:
                basis_indices = out_dict[i]["multi_indices"]

                # Evaluate the multivariate polynomials on X
                psi = create_psi(basis_indices, _univ_p_val)

                # Calulate the cofficients of surrogate model
                updated_out = self.regression(
                    psi, y[:, i], basis_indices, sparsity=False
                )

                # Update coeffs in out_dict
                final_out_dict[i]["coeffs"][nnz_idx] = updated_out["coeffs"]

        return final_out_dict

    # -------------------------------------------------------------------------

    def univ_basis_vals(self, samples, n_max=None):
        """
        Evaluates univariate regressors along input directions.

        Parameters
        ----------
        samples : array of shape (n_samples, ndim)
            Samples.
        n_max : int, optional
            Maximum polynomial degree. The default is `None`.

        Returns
        -------
        univ_basis: array of shape (n_samples, ndim, n_max+1)
            All univariate regressors up to n_max.
        """
        # Extract information
        poly_types = self.input_space.poly_types
        if samples.ndim != 2:
            samples = samples.reshape(1, len(samples))
        n_max = np.max(self.pce_deg) if n_max is None else n_max

        # Extract poly coeffs
        if self.input_space.input_data_given or self.meta_model_type.lower() == "apce":
            a_polycoeffs = self._polycoeffs
        else:
            a_polycoeffs = None

        # Evaluate univariate basis
        univ_basis = eval_univ_basis(samples, n_max, poly_types, a_polycoeffs)

        return univ_basis

    # -------------------------------------------------------------------------

    def regression(self, X, y, basis_indices, sparsity=True):
        """
        Fit regression using the regression method provided.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array of shape (n_samples,)
            Target values.
        basis_indices : array of shape (n_terms, ndim)
            Multi-indices of multivariate polynomials.
        reg_method : str, optional
            DESCRIPTION. The default is None.
        sparsity : bool
            Use with sparsity-inducing training methods. The default is True

        Returns
        -------
        return_out_dict : Dict
            Fitted estimator, spareMulti-Index, sparseX and coefficients.

        """
        # Check for valid method
        reg_method = self.pce_reg_method.lower()
        if reg_method not in self.regression_dict:
            raise AttributeError("The set regression method is not available.")

        # Bayes sparse adaptive aPCE
        _clf_poly = None
        kwargs = self._pce_reg_options[reg_method]

        # Add the last parameters
        # NEW: Added fastardextended
        if reg_method in ["brr", "ard", "fastard", "fastardextended"]:
            compute_score = bool(self.verbose)
            kwargs["compute_score"] = compute_score

        if reg_method in ["brr", "ard"]:
            lambda_ = 1e-6
            if np.var(y) != 0:
                lambda_ = 1 / np.var(y)
            kwargs["lambda_1"] = lambda_
            kwargs["lambda_2"] = lambda_

        if reg_method == "bcs":
            bias_term = self.dim_red_method.lower() != "pca"
            kwargs["bias_term"] = bias_term

        # Handle any exceptions
        if reg_method == "ard":
            if X.shape[0] < 2:
                raise ValueError(
                    "Regression with ARD can only be performed for more than 2 samples"
                )
        if reg_method == "bcs":
            if X.shape[0] < 10:
                raise ValueError(
                    "Regression with BCS can only be performed for more than 10 samples"
                )
        if reg_method == "lassolars":
            if X.shape[0] < 10:
                raise ValueError(
                    "Regression with LARS can only be performed for more than 5 samples"
                )

        # Init the regression object
        _clf_poly = self.regression_dict[reg_method](**kwargs)

        # Apply any other settings
        if reg_method == "brr":
            _clf_poly.converged = True

        # Do the fit
        _clf_poly.fit(X, y)

        # Select the nonzero entries of coefficients
        if sparsity:
            nnz_idx = np.nonzero(_clf_poly.coef_)[0]
        else:
            nnz_idx = np.arange(_clf_poly.coef_.shape[0])

        # This is for the case where all outputs are zero, thereby
        # all coefficients are zero
        if (y == 0).all():
            nnz_idx = np.insert(np.nonzero(_clf_poly.coef_)[0], 0, 0)
        # If all coefficients are zero due to other reasons apply the same fix.
        if len(nnz_idx) == 0:
            nnz_idx = np.insert(np.nonzero(_clf_poly.coef_)[0], 0, 0)

        # Get sparse basis indices and PSI matrix
        sparse_basis_indices = basis_indices[nnz_idx]
        sparse_x = X[:, nnz_idx]

        # Check if re-estimation of coefficients is necessary:
        if reg_method in ["lars", "lassolars", "lassolarscv"] and nnz_idx.shape[0] < _clf_poly.coef_.shape[0]:
            # Re-estimate PCE coefficients using OLS
            _clf_poly = lm.LinearRegression(fit_intercept=False)
            _clf_poly.fit(sparse_x, y)
            coeffs = _clf_poly.coef_
        else:
            # If no re-estimation is needed, extract the sparse coefficients from original solver
            coeffs = _clf_poly.coef_[nnz_idx]
            _clf_poly.coef_ = coeffs

        # Create a dict to pass the outputs
        return_out_dict = {}
        return_out_dict["clf_poly"] = _clf_poly
        return_out_dict["spareMulti-Index"] = sparse_basis_indices
        return_out_dict["sparePsi"] = sparse_x
        return_out_dict["coeffs"] = coeffs
        return return_out_dict

    # -------------------------------------------------------------------------

    def adaptive_regression(self, X, y, var_idx, verbose=False):
        """
        Adaptively fits the PCE model by comparing the scores of different
        degrees and q-norm.

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
            Fitted estimator, best degree, best q-norm, loo_cv_score and
            coefficients.

        """

        # Initialization
        q_all_coeffs, all_coeffs = {}, {}
        q_all_indices_sparse, all_indices_sparse = {}, {}
        q_all_clf_poly, all_clf_poly = {}, {}
        q_all_in_terms, all_in_terms = {}, {}
        q_all_lc_error, all_lc_error = {}, {}

        # Evaluate the univariate polynomials
        _univ_p_val = self.univ_basis_vals(X)

        # Extract degree array and q-norm array
        _deg_array = np.array([*self._all_basis_indices], dtype=int)
        qnorm = [*self._all_basis_indices[str(int(_deg_array[0]))]]

        # Some options for EarlyStop
        error_increases = False
        # Stop degree, if LOO error does not decrease n_checks_degree times
        n_checks_degree = 3
        # Stop qNorm, if criterion isn't fulfilled n_checks_q_norm times
        n_checks_q_norm = 2
        n_qnorms = len(qnorm)
        q_norm_early_stop = True
        if n_qnorms < n_checks_q_norm + 1:
            q_norm_early_stop = False

        # =====================================================================
        # basis adaptive polynomial chaos: repeat the calculation by increasing
        # polynomial degree until the highest accuracy is reached
        # =====================================================================
        # For each degree check all q-norms and choose the best one
        best_q = None
        scores = -np.inf * np.ones(_deg_array.shape[0])
        q_norm_scores = -np.inf * np.ones(n_qnorms)

        for deg_idx, deg in enumerate(_deg_array):

            for qidx, q in enumerate(qnorm):

                # Extract the polynomial basis indices from the pool of
                # _all_basis_indices
                basis_indices = self._all_basis_indices[str(deg)][str(q)]

                # Assemble the Psi matrix
                psi = create_psi(basis_indices, _univ_p_val)

                # Calulate the cofficients of the metamodel
                outs = self.regression(psi, y, basis_indices)

                # Calculate and save the score of LOOCV
                score, lc_error = corr_loocv_error(
                    outs["clf_poly"], outs["sparePsi"], outs["coeffs"], y
                )

                # Check the convergence of noise for FastARD
                if (
                    self.pce_reg_method == "FastARD"
                    and outs["clf_poly"].alpha_ < np.finfo(np.float32).eps
                ):
                    score = -np.inf

                q_norm_scores[qidx] = score
                q_all_coeffs[str(qidx + 1)] = outs["coeffs"]
                q_all_indices_sparse[str(qidx + 1)] = outs["spareMulti-Index"]
                q_all_clf_poly[str(qidx + 1)] = outs["clf_poly"]
                q_all_in_terms[str(qidx + 1)] = basis_indices.shape[0]
                q_all_lc_error[str(qidx + 1)] = lc_error

                # EarlyStop check
                # if there are at least n_checks_q_norm entries after the
                # best one, we stop
                if (
                    q_norm_early_stop
                    and sum(np.isfinite(q_norm_scores)) > n_checks_q_norm
                ):
                    # If the error has increased the last two iterations, stop!
                    q_norm_scores_noninf = q_norm_scores[np.isfinite(q_norm_scores)]
                    deltas = np.sign(np.diff(q_norm_scores_noninf))
                    if sum(deltas[-n_checks_q_norm + 1 :]) == 2:
                        # stop the q-norm loop here
                        break
                if np.var(y) == 0:
                    break

            # Store the score in the scores list
            try:
                best_q = np.nanargmax(q_norm_scores)
            # Workaround for All-NaN slices
            except ValueError:
                best_q = q_norm_scores.shape[0] - 1
            scores[deg_idx] = q_norm_scores[best_q]

            all_coeffs[str(deg_idx + 1)] = q_all_coeffs[str(best_q + 1)]
            all_indices_sparse[str(deg_idx + 1)] = q_all_indices_sparse[str(best_q + 1)]
            all_clf_poly[str(deg_idx + 1)] = q_all_clf_poly[str(best_q + 1)]
            all_in_terms[str(deg_idx + 1)] = q_all_in_terms[str(best_q + 1)]
            all_lc_error[str(deg_idx + 1)] = q_all_lc_error[str(best_q + 1)]

            # Check the direction of the error (on average):
            # if it increases consistently stop the iterations
            if len(scores[scores != -np.inf]) > n_checks_degree:
                scores_noninf = scores[scores != -np.inf]
                ss = np.sign(scores_noninf - np.max(scores_noninf))
                # ss<0 error decreasing
                error_increases = np.sum(np.sum(ss[-2:])) <= -1 * n_checks_degree

            if error_increases:
                break

            # Check only one degree, if target matrix has zero variance
            if np.var(y) == 0:
                break

        # ------------------ Summary of results ------------------
        # Select the one with the best score and save the necessary outputs
        best_deg = np.nanargmax(scores) + 1
        coeffs = all_coeffs[str(best_deg)]
        basis_indices = all_indices_sparse[str(best_deg)]
        _clf_poly = all_clf_poly[str(best_deg)]
        loo_cv_score = np.nanmax(scores)
        p = all_in_terms[str(best_deg)]
        lc_error = all_lc_error[str(best_deg)]
        degree = _deg_array[np.nanargmax(scores)]
        qnorm = float(qnorm[best_q])

        # ------------------ Print out Summary of results ------------------
        if self.verbose:
            # Create PSI_Sparse by removing redundent terms
            nnz_idx = np.nonzero(coeffs)[0]
            basis_indices_sparse = basis_indices[nnz_idx]

            print(f"Output variable {var_idx + 1}:")
            print(
                "The estimation of PCE coefficients converged at polynomial "
                f"degree {_deg_array[best_deg - 1]} with "
                f"{len(basis_indices_sparse)} terms (Sparsity index = "
                f"{round(len(basis_indices_sparse) / p, 3)})."
            )

            print(f"Final ModLOO error estimate: {1 - max(scores):.3e}")
            print("\n" + "-" * 50)

        if verbose:
            print("=" * 50)
            print(" " * 10 + " Summary of results ")
            print("=" * 50)

            print("Scores:\n", scores)
            print("Degree of best score:", self._deg_array[best_deg - 1])
            print("No. of terms:", len(basis_indices))
            print("Sparsity index:", round(len(basis_indices) / p, 3))
            print("Best Indices:\n", basis_indices)

            if self.pce_reg_method.lower() in ["brr", "ard"]:
                _, ax = plt.subplots(figsize=(12, 10))
                plt.title("Marginal log-likelihood")
                plt.plot(_clf_poly.scores_, color="navy", linewidth=2)
                plt.ylabel("Score")
                plt.xlabel("Iterations")
                if self.pce_reg_method.lower() == "bbr":
                    text = f"$\\alpha={_clf_poly.alpha_:.1f}$\n\
                        $\\lambda={_clf_poly.lambda_:.3f}$\n\
                        $L={_clf_poly.scores_[-1]:.1f}$"
                else:
                    text = f"$\\alpha={_clf_poly.alpha_:.1f}$\n$\
                        \\L={_clf_poly.scores_[-1]:.1f}$"

                plt.text(0.75, 0.5, text, fontsize=18, transform=ax.transAxes)
                plt.savefig(f"marg_loglik_{self.pce_reg_method}.png")
                plt.close()
            print("=" * 80)

        # Create a dict to pass the outputs
        return_vars = {}
        return_vars["clf_poly"] = _clf_poly
        return_vars["degree"] = degree
        return_vars["qnorm"] = qnorm
        return_vars["coeffs"] = coeffs
        return_vars["multi_indices"] = basis_indices
        return_vars["loo_cv_score"] = loo_cv_score
        return_vars["lc_error"] = lc_error

        return return_vars

    # -------------------------------------------------------------------------

    @_preprocessing_eval
    @_bootstrap_eval
    def eval_metamodel(self, samples, b_i=0, parallel=False):
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
            Standard deviatioon of the predictions.
        """
        # Compute univariate bases for the given samples
        _univ_p_val = self.univ_basis_vals(samples, n_max=np.max(self.pce_deg))

        # Extract model dictionary
        model_dict = self._coeffs_dict[f"b_{b_i + 1}"]

        # Loop over outputs
        mean_pred = {}
        std_pred = {}
        for output, values in model_dict.items():
            mean = np.empty((len(samples), len(values)))
            std = np.empty((len(samples), len(values)))
            idx = 0
            for in_key, _ in values.items():

                # Assemble Psi matrix
                basis = self._basis_dict[f"b_{b_i + 1}"][output][in_key]
                psi = create_psi(basis, _univ_p_val)

                # Prediction
                if self.bootstrap_method != "fast" or b_i == 0:
                    # with error bar, i.e. use _clf_poly
                    _clf_poly = self._clf_poly[f"b_{b_i + 1}"][output][in_key]
                    try:
                        y_mean, y_std = _clf_poly.predict(psi, return_std=True)
                    except TypeError:
                        y_mean = _clf_poly.predict(psi)
                        y_std = np.zeros_like(y_mean)
                else:
                    # without error bar
                    coeffs = self._coeffs_dict[f"b_{b_i + 1}"][output][in_key]
                    y_mean = np.dot(psi, coeffs)
                    y_std = np.zeros_like(y_mean)

                mean[:, idx] = y_mean
                std[:, idx] = y_std
                idx += 1

            mean_pred[output] = mean
            std_pred[output] = std

        return mean_pred, std_pred

    # -------------------------------------------------------------------------
    def _select_degree(self):  # , ndim, n_samples):
        """
        Selects degree based on the number of samples and parameters in the
        sequential design.

        Parameters
        ----------
        ndim : int
            Dimension of the parameter space.
        n_samples : int
            Number of samples.

        Returns
        -------
        _deg_array: array
            The selected degrees.

        """
        # Define the _deg_array
        max_deg = np.max(self.pce_deg)
        min_deg = np.min(self.pce_deg)

        # The old options for sequential are commented out!!!
        # nitr = n_samples - self.input_space.n_init_samples

        # Check q-norm
        if not np.isscalar(self.pce_q_norm):
            self.pce_q_norm = np.array(self.pce_q_norm)
        else:
            self.pce_q_norm = np.array([self.pce_q_norm])

        # def M_uptoMax(maxDeg):
        #    n_combo = np.zeros(maxDeg)
        #    for i, d in enumerate(range(1, maxDeg + 1)):
        #        n_combo[i] = math.factorial(ndim + d)
        #        n_combo[i] /= math.factorial(ndim) * math.factorial(d)
        #    return n_combo

        deg_new = max_deg
        # d = nitr if nitr != 0 and self.ndim > 5 else 1
        # d = 1
        # min_index = np.argmin(abs(M_uptoMax(max_deg)-ndim*n_samples*d))
        # deg_new = range(1, max_deg+1)[min_index]

        if deg_new > min_deg and self.pce_reg_method.lower() != "fastard":
            _deg_array = np.arange(min_deg, deg_new + 1)
        else:
            _deg_array = np.array([deg_new])

        return _deg_array

    def _generate_polynomials(self, max_deg=None):
        """
        Generates (univariate) polynomials.

        Parameters
        ----------
        max_deg : int
            Maximum polynomial degree.

        Returns
        -------
        None
        """
        # Check for input_space
        if self.input_space is None:
            raise AttributeError(
                "Generate or add InputSpace before generating polynomials"
            )

        ndim = self.input_space.ndim
        # Create orthogonal polynomial coefficients if necessary
        if max_deg is not None:  # and self.input_obj.poly_coeffs_flag:
            self._polycoeffs = {}
            if self.verbose:
                for par_idx in tqdm(
                    range(ndim), ascii=True, desc="Computing orth. polynomial coeffs"
                ):
                    poly_coeffs = apoly_construction(
                        self.input_space.raw_data[par_idx], max_deg
                    )
                    self._polycoeffs[f"p_{par_idx + 1}"] = poly_coeffs
            else:
                for par_idx in range(ndim):
                    poly_coeffs = apoly_construction(
                        self.input_space.raw_data[par_idx], max_deg
                    )
                    self._polycoeffs[f"p_{par_idx + 1}"] = poly_coeffs

        else:
            raise AttributeError(
                "MetaModel cannot generate polynomials in the given scenario!"
            )

    # -------------------------------------------------------------------------
    def calculate_moments(self):
        """
        Computes the first two moments using the PCE-based metamodel.

        Returns
        -------
        pce_means: dict
            The first moment (mean) of the surrogate.
        pce_stds: dict
            The second moment (standard deviation) of the surrogate.

        """
        outputs = self.out_names
        pce_means_b = {}
        pce_stds_b = {}

        for b_i in range(self.n_bootstrap_itrs):
            pce_means_b[b_i] = {}
            pce_stds_b[b_i] = {}
            _coeffs_dicts = self._coeffs_dict[f"b_{b_i + 1}"].items()

            for output, coef_dict in _coeffs_dicts:
                pce_mean = np.zeros((len(coef_dict)))
                pce_var = np.zeros((len(coef_dict)))

                for index, _ in coef_dict.items():
                    idx = int(index.split("_")[1]) - 1
                    coeffs = self._coeffs_dict[f"b_{b_i + 1}"][output][index]

                    if coeffs[0] != 0:
                        pce_mean[idx] = coeffs[0]
                    else:
                        _clf_poly = self._clf_poly[f"b_{b_i + 1}"][output]
                        pce_mean[idx] = _clf_poly[index].intercept_
                    pce_var[idx] = np.sum(np.square(coeffs[1:]))

                # Save predictions for each output
                if self.dim_red_method.lower() == "pca":
                    pca = self.pca[f"b_{b_i + 1}"][output]
                    pce_means_b[b_i][output] = pca.inverse_transform(pce_mean)
                    pce_stds_b[b_i][output] = pca.inverse_transform(np.sqrt(pce_var))
                else:
                    pce_means_b[b_i][output] = pce_mean
                    pce_stds_b[b_i][output] = np.sqrt(pce_var)

        # Change the order of nesting
        mean_all = {}
        for i in sorted(pce_means_b):
            for k, v in pce_means_b[i].items():
                if k not in mean_all:
                    mean_all[k] = [None] * len(pce_means_b)
                mean_all[k][i] = v
        std_all = {}
        for i in sorted(pce_stds_b):
            for k, v in pce_stds_b[i].items():
                if k not in std_all:
                    std_all[k] = [None] * len(pce_stds_b)
                std_all[k][i] = v

        # Back transformation if PCA is selected.
        pce_means, pce_stds = {}, {}
        for output in outputs:
            pce_means[output] = np.mean(mean_all[output], axis=0)
            pce_stds[output] = np.mean(std_all[output], axis=0)

        # Print a report table
        if self.verbose:
            for output in outputs:
                print(f"\n>>>>> Moments of {output} <<<<<")
                print("\nIndex  |  Mean   |  Std. deviation")
                print("-" * 35)
                print(
                    "\n".join(
                        f"{i+1}  |  {k:.3e}  |  {j:.3e}"
                        for i, (k, j) in enumerate(
                            zip(pce_means[output], pce_stds[output])
                        )
                    )
                )
            print("-" * 40)

        return pce_means, pce_stds

    def calculate_sobol_old(self, y_train=None):
        """
        Provides Sobol' indices as a sensitivity measure to infer the importance
        of the input parameters. See Eq. 27 in [1] for more details. For the
        case with Principal component analysis refer to [2].

        [1] Global sensitivity analysis: A flexible and efficient framework
        with an example from stochastic hydrogeology S. Oladyshkin, F.P.
        de Barros, W. Nowak  https://doi.org/10.1016/j.advwatres.2011.11.001

        [2] Nagel, J.B., Rieckermann, J. and Sudret, B., 2020. Principal
        component analysis and sparse polynomial chaos expansions for global
        sensitivity analysis and model calibration: Application to urban
        drainage simulation. Reliability Engineering & System Safety, 195,
        p.106737.

        Parameters
        ----------
        y_train: dict, optional
            Trainings outputs. They are needed when used in combination with PCA.
            The default is None

        Returns
        -------
        sobol : dict
            Sobol' indices of all available orders.
        totalsobol : dict
            Total Sobol' indices

        """
        if self.dim_red_method.lower() == "pca" and y_train is None:
            raise AttributeError(
                "Calculation of Sobol' indices with PCA expects training outputs,"
                + " but none are given."
            )

        max_order = np.max(self.pce_deg)
        n_params = len(self.input_obj.marginals)
        cov_zpq = np.zeros((n_params))
        outputs = self.out_names
        sobol_cell_b, total_sobol_b = {}, {}

        for b_i in range(self.n_bootstrap_itrs):
            sobol_cell_, total_sobol_ = {}, {}

            for output in outputs:
                n_meas_points = len(self._coeffs_dict[f"b_{b_i+1}"][output])

                # Initialize the (cell) array containing the (total) Sobol indices.
                sobol_array = dict.fromkeys(range(1, max_order + 1), [])
                sobol_cell_array = dict.fromkeys(range(1, max_order + 1), [])

                for i_order in range(1, max_order + 1):
                    n_comb = math.comb(n_params, i_order)
                    sobol_cell_array[i_order] = np.zeros((n_comb, n_meas_points))
                total_sobol_array = np.zeros((n_params, n_meas_points))

                # Initialize the cell to store the names of the variables
                total_variance = np.zeros((n_meas_points))
                for p_idx in range(n_meas_points):
                    basis = self._basis_dict[f"b_{b_i+1}"][output][f"y_{p_idx+1}"]
                    coeffs = self._coeffs_dict[f"b_{b_i+1}"][output][f"y_{p_idx+1}"]

                    # Compute total variance
                    total_variance[p_idx] = np.sum(np.square(coeffs[1:]))

                    nzidx = np.where(coeffs != 0)[0]
                    # Set all the Sobol indices equal to zero in the presence of a
                    # null output.
                    if len(nzidx) == 0:
                        for i_order in range(1, max_order + 1):
                            sobol_cell_array[i_order][:, p_idx] = 0

                    # Otherwise compute them from sums of the coefficients
                    else:
                        nz_basis = basis[nzidx]
                        for i_order in range(1, max_order + 1):
                            idx = np.where(np.sum(nz_basis > 0, axis=1) == i_order)
                            subbasis = nz_basis[idx]
                            z_ = np.array(list(combinations(range(n_params), i_order)))

                            for q in range(z_.shape[0]):
                                if total_variance[p_idx] == 0.0:
                                    sobol_cell_array[i_order][q, p_idx] = 0.0
                                else:
                                    subidx = np.prod(subbasis[:, z_[q]], axis=1) > 0
                                    sum_ind = nzidx[idx[0][subidx]]
                                    sobol = np.sum(np.square(coeffs[sum_ind]))
                                    sobol /= total_variance[p_idx]
                                    sobol_cell_array[i_order][q, p_idx] = sobol

                        # Compute the TOTAL Sobol indices.
                        for par_idx in range(n_params):
                            idx = nz_basis[:, par_idx] > 0
                            sum_ind = nzidx[idx]

                            if total_variance[p_idx] == 0.0:
                                total_sobol_array[par_idx, p_idx] = 0.0
                            else:
                                sobol = np.sum(np.square(coeffs[sum_ind]))
                                sobol /= total_variance[p_idx]
                                total_sobol_array[par_idx, p_idx] = sobol

                    # ----- if PCA selected: Compute covariance -----
                    if self.dim_red_method.lower() == "pca":
                        # Extract the basis indices (alpha) and coefficients for
                        # next component
                        if p_idx < n_meas_points - 1:
                            nextbasis = self._basis_dict[f"b_{b_i+1}"][output][
                                f"y_{p_idx+2}"
                            ]
                            if self.bootstrap_method != "fast" or b_i == 0:
                                clf_poly = self._clf_poly[f"b_{b_i+1}"][output][
                                    f"y_{p_idx+2}"
                                ]
                                next_coeffs = clf_poly.coef_
                            else:
                                next_coeffs = self._coeffs_dict[f"b_{b_i+1}"][output][
                                    f"y_{p_idx+2}"
                                ]

                            # Choose the common non-zero basis
                            mask = (basis[:, None] == nextbasis).all(-1).any(-1)
                            n_mask = (nextbasis[:, None] == basis).all(-1).any(-1)

                            # Compute the covariance in Eq 17.
                            for par_idx in range(n_params):
                                idx = (mask) & (basis[:, par_idx] > 0)
                                n_idx = (n_mask) & (nextbasis[:, par_idx] > 0)
                                try:
                                    cov_zpq[par_idx] += np.sum(
                                        np.dot(coeffs[idx], next_coeffs[n_idx])
                                    )
                                except:
                                    pass

                # Compute the sobol indices according to Ref. 2
                if self.dim_red_method.lower() == "pca":
                    n_c_points = y_train[output].shape[1]
                    pca = self.pca[f"b_{b_i+1}"][output]
                    comp_pca = pca.components_
                    n_comp = comp_pca.shape[0]
                    var_zp = pca.explained_variance_

                    # Extract the sobol index of the components
                    for i_order in range(1, max_order + 1):
                        n_comb = math.comb(n_params, i_order)
                        sobol_array[i_order] = np.zeros((n_comb, n_c_points))
                        z_ = np.array(list(combinations(range(n_params), i_order)))

                        # Loop over parameters
                        for q in range(z_.shape[0]):
                            s_zi = sobol_cell_array[i_order][q]

                            for t_idx in range(n_c_points):
                                var_yt = np.var(y_train[output][:, t_idx])
                                term1, term2 = 0.0, 0.0
                                if var_yt != 0.0:
                                    # Eq. 17
                                    for i in range(n_comp):
                                        a = s_zi[i] * var_zp[i]
                                        a *= comp_pca[i, t_idx] ** 2
                                        term1 += a

                                    # Term 2
                                    # term2 = 0.0
                                    # for i in range(nComp-1):
                                    #     term2 += cov_Z_p_q[q] * compPCA[i, tIdx]
                                    #     term2 *= compPCA[i+1, tIdx]
                                    # term2 *= 2

                                # Divide over total output variance Eq. 18
                                sobol_array[i_order][q, t_idx] = term1  # + term2
                                sobol_array[i_order][q, t_idx] /= var_yt

                    # Compute the TOTAL Sobol indices.
                    total_sobol = np.zeros((n_params, n_c_points))
                    for par_idx in range(n_params):
                        s_zi = total_sobol_array[par_idx]

                        for t_idx in range(n_c_points):
                            var_yt = np.var(y_train[output][:, t_idx])
                            term1, term2 = 0.0, 0.0
                            if var_yt != 0.0:
                                for i in range(n_comp):
                                    term1 += (
                                        s_zi[i] * var_zp[i] * (comp_pca[i, t_idx] ** 2)
                                    )
                                for i in range(n_comp - 1):
                                    term2 += (
                                        cov_zpq[par_idx]
                                        * comp_pca[i, t_idx]
                                        * comp_pca[i + 1, t_idx]
                                    )
                                term2 *= 2

                            total_sobol[par_idx, t_idx] = term1  # + term2

                            # Devide over total output variance Eq. 18
                            total_sobol[par_idx, t_idx] /= var_yt

                    sobol_cell_[output] = sobol_array
                    total_sobol_[output] = total_sobol
                else:
                    sobol_cell_[output] = sobol_cell_array
                    total_sobol_[output] = total_sobol_array

            # Save for each bootstrap iteration
            sobol_cell_b[b_i] = sobol_cell_
            total_sobol_b[b_i] = total_sobol_

        # Combine for Sobol' indices, one set of indices for each possible degree of interaction
        self.sobol = {}
        sobol_all = {}
        for i in sorted(sobol_cell_b):
            for k, v in sobol_cell_b[i].items():
                for l, _ in v.items():
                    if l not in sobol_all:
                        sobol_all[l] = {}
                    if k not in sobol_all[l]:
                        sobol_all[l][k] = [None] * len(sobol_cell_b)
                    sobol_all[l][k][i] = v[l]
        for i_order in range(1, max_order + 1):
            self.sobol[i_order] = {}
            for output in outputs:
                self.sobol[i_order][output] = np.mean(
                    [sobol_all[i_order][output]], axis=0
                )

        # Combine for Total Sobol' indices
        total_sobol_all = {}
        self.total_sobol = {}
        for i in sorted(total_sobol_b):
            for k, v in total_sobol_b[i].items():
                if k not in total_sobol_all:
                    total_sobol_all[k] = [None] * len(total_sobol_b)
                total_sobol_all[k][i] = v
        for output in outputs:
            self.total_sobol[output] = np.mean(total_sobol_all[output], axis=0)

        print(self.sobol)
        print(self.total_sobol)

        return self.sobol, self.total_sobol

    def calculate_sobol(self, y_train=None):
        """
        Calculates S1 and ST for all parameters across all outputs, 
        bootstrap iterations, and time points, accounting for time-varying bases.
        """
        
        sobol_results = {}

        for out_key in self.out_names:
            first_b_key = "b_1"
            n_boot = self.n_bootstrap_itrs
            n_meas_points = len(self._coeffs_dict[first_b_key][out_key])
            n_params = len(self.par_names)

            # Initialize result containers: Shape (n_boot, n_params, n_meas_points)
            # We use arrays here for easier indexing before converting to dicts
            S1_cube = np.zeros((n_boot, n_params, n_meas_points))
            ST_cube = np.zeros((n_boot, n_params, n_meas_points))

            # We must iterate through time points because the basis is dynamic
            for t_i in range(n_meas_points):
                t_str = f"y_{t_i+1}"
                
                # 1. Get the basis matrix for THIS specific time point
                # Assuming basis is consistent across bootstrap iterations for the same t_i
                basis_matrix = np.array(self._basis_dict[first_b_key][out_key][t_str])
                n_coeffs = basis_matrix.shape[0]

                # 2. Extract Coefficients for all bootstrap samples at this time point
                # Shape: (n_boot, n_coeffs)
                C_t = np.zeros((n_boot, n_coeffs))
                for b_i in range(n_boot):
                    b_str = f"b_{b_i+1}"
                    C_t[b_i, :] = self._coeffs_dict[b_str][out_key][t_str]

                # 3. Calculate Total Variance for this time point (Denominator)
                # Skip index 0 (intercept)
                total_var_t = np.sum(np.square(C_t[:, 1:]), axis=1)
                total_var_t[total_var_t == 0] = np.nan 

                # 4. Compute Indices per parameter for this time point
                for p_idx, p_name in enumerate(self.par_names):
                    # First Order Mask: Interaction order is 1 AND parameter is present
                    s1_mask = (np.sum(basis_matrix > 0, axis=1) == 1) & (basis_matrix[:, p_idx] > 0)
                    
                    # Total Order Mask: Parameter is involved in the term
                    st_mask = (basis_matrix[:, p_idx] > 0)

                    # Sum squares of coeffs and normalize across the bootstrap axis
                    S1_cube[:, p_idx, t_i] = np.sum(np.square(C_t[:, s1_mask]), axis=1) / total_var_t
                    ST_cube[:, p_idx, t_i] = np.sum(np.square(C_t[:, st_mask]), axis=1) / total_var_t

            # 5. Reformat into the requested dictionary structure
            s1_map = {p_name: S1_cube[:, i, :] for i, p_name in enumerate(self.par_names)}
            st_map = {p_name: ST_cube[:, i, :] for i, p_name in enumerate(self.par_names)}
            
            sobol_results[out_key] = {"S1": s1_map, "ST": st_map}

        return sobol_results

    def calculate_sobol_current_version(self, y_train=None):
        """
        # Warning: this was reduced to only the most basic version of the calculation
        # - no pca

        Provides Sobol' indices as a sensitivity measure to infer the importance
        of the input parameters. See Eq. 27 in [1] for more details. For the
        case with Principal component analysis refer to [2].

        [1] Global sensitivity analysis: A flexible and efficient framework
        with an example from stochastic hydrogeology S. Oladyshkin, F.P.
        de Barros, W. Nowak  https://doi.org/10.1016/j.advwatres.2011.11.001

        [2] Nagel, J.B., Rieckermann, J. and Sudret, B., 2020. Principal
        component analysis and sparse polynomial chaos expansions for global
        sensitivity analysis and model calibration: Application to urban
        drainage simulation. Reliability Engineering & System Safety, 195,
        p.106737.

        Parameters
        ----------
        y_train: dict, optional
            Trainings outputs. They are needed when used in combination with PCA.
            The default is None

        Returns
        -------
        sobol : dict
            Sobol' indices of all available orders.
            Format: stacked dictionary with inner np.array
            key order: {sobol_order {parameters {out_key [#bootstrap, #x_values]}}}
        totalsobol : dict
            Total Sobol' indices

        """
        if self.dim_red_method.lower() == "pca" and y_train is None:
            raise AttributeError(
                "Calculation of Sobol' indices with PCA expects training outputs,"
                + " but none are given."
            )

        # Only calculate up to where sensible
        max_order = min(np.max(self.pce_deg), len(self.par_names))
        param_list = calc_param_combinations(self.par_names)
        n_params = len(self.input_obj.marginals)
        n_meas_points = len(self._coeffs_dict["b_1"][self.out_names[0]])

        # Init output array
        i_names = [f"Sobol_{i+1}" for i in range(max_order)]
        out_dict = {
            key: np.zeros((self.n_bootstrap_itrs, n_meas_points))
            for key in self.out_names
        }
        indices = {
            name: {
                str(param): copy.deepcopy(out_dict) for param in param_list[n_id + 1]
            }
            for n_id, name in enumerate(i_names)
        }

        i_names.append("TotalSobol")
        indices["TotalSobol"] = {
            str(param): copy.deepcopy(out_dict) for param in param_list[1]
        }

        # Sobol indices
        for i in range(max_order):
            order = i + 1
            s_key = f"Sobol_{order}"

            z_ = np.array(list(combinations(range(n_params), order)))

            for p_id, pars in enumerate(param_list[order]):
                for out_key in self.out_names:
                    for b_i in range(self.n_bootstrap_itrs):
                        for t in range(n_meas_points):
                            # Setups
                            basis = self._basis_dict[f"b_{b_i+1}"][out_key][f"y_{t+1}"]
                            coeffs = self._coeffs_dict[f"b_{b_i+1}"][out_key][
                                f"y_{t+1}"
                            ]
                            tot_var = np.sum(np.square(coeffs[1:]))
                            nzidx = np.where(coeffs != 0)[0]

                            # If all coeffs 0 - index is 0
                            if len(nzidx) == 0:
                                continue
                            # If total variance 0 - index is 0
                            if tot_var == 0:
                                continue

                            # Calculate Sobol' index
                            nz_basis = basis[nzidx]
                            idx = np.where(np.sum(nz_basis > 0, axis=1) == order)
                            subbasis = nz_basis[idx]

                            subidx = np.prod(subbasis[:, z_[p_id]], axis=1) > 0
                            sum_ind = nzidx[idx[0][subidx]]
                            s_idx = np.sum(np.square(coeffs[sum_ind])) / tot_var
                            indices[s_key][str(pars)][out_key][b_i, t] = s_idx

                            # Add to Total index
                            for par_id2, par in enumerate(self.par_names):
                                if par_id2 in z_[p_id]:
                                    indices["TotalSobol"][str([par])][out_key][
                                        b_i, t
                                    ] += s_idx

        return indices

    def build_der_basis_coeffs(self, deriv_num, der_basis_coeffs):
        """
        Recursive calculation of coefficients of the derivative univariate
        basis for all input parameters.

        Parameters
        ----------
        deriv_num : int
            Degree of the derivative.
        der_basis_coeffs : dict
            Previously calculated derivative coefficients. If no derivative has
            been calculated, it should contain the univariate coefficients
            under the key '0'.

        Returns
        -------
        der_basis_coeffs : dict
            Updated dictionary of coefficients up to derivative degree deriv_num.

        """
        # If the previous derivative does not exist, go back one
        if (deriv_num - 1 > 0) and ((deriv_num - 1) not in der_basis_coeffs):
            der_basis_coeffs = self.build_der_basis_coeffs(
                deriv_num - 1, der_basis_coeffs
            )
        # If the previous derivative exists, build the current one
        acoeffs = der_basis_coeffs[deriv_num - 1]
        der_basis_coeffs[deriv_num] = copy.deepcopy(acoeffs)
        # Inner derivatives of the univariate polynomials
        for param in range(len(self.par_names)):
            der_basis = der_basis_coeffs[deriv_num][f"p_{param+1}"]
            aco = acoeffs[f"p_{param+1}"]
            for d in range(aco.shape[0]):
                for j in range(aco.shape[1]):
                    der_basis[d, j - 1] = aco[d, j] * j
                    if j == 0:
                        der_basis[d, j - 1] *= 0
            der_basis_coeffs[deriv_num][f"p_{param+1}"] = der_basis
        return der_basis_coeffs

    def build_deriv_inner_deg(self, deriv_names):
        """
        Calculation of the degrees of the derivative univariate bases for each
        parameter and derivative degree in deriv_names.

        Parameters
        ----------
        deriv_names : dict
            Description of the derivative. This includes name of the input
            parameters to use for the derivatives as dictionary keys, and
            the degree of derivative wrt the parameter as the value.

        Returns
        -------
        None.

        """
        for par_name in deriv_names:
            par_num = self.par_names.index(par_name)
            degree = deriv_names[par_name]

            # Calculate all the degrees and coeffs if not there
            if self.deriv_inner_deg[degree][par_name] == None:
                degrees = self._basis_dict["b_1"]
                inner_degrees = copy.deepcopy(degrees)
                for key in list(degrees.keys()):
                    for times in list(degrees[key].keys()):
                        for i in range(degrees[key][times].shape[0]):
                            deg = degrees[key][times][i]
                            inner_degrees[key][times][i] = np.zeros(deg.shape[0])
                            # Set inner degree if the coefficient exists
                            if deg[par_num] != 0:
                                inner_degrees[key][times][i][par_num] = deg[par_num]
                self.deriv_inner_deg[degree][par_name] = inner_degrees

    def build_outer_deriv(self, deriv_names):
        """
        Build the components for the outer derivative.
        This is done by setting the degrees to 0 for all parameters involved
        in the derivatives, and setting the multiplier-coefficients to 1 for
        terms that involve all derivative-parameters.

        Parameters
        ----------
        deriv_names : dict
            Description of the derivative. This includes name of the input
            parameters to use for the derivatives as dictionary keys, and
            the degree of derivative wrt the parameter as the value.

        Returns
        -------
        None.

        """
        deriv_string = json.dumps(deriv_names)

        # Calculate all the degrees and coeffs if not there
        if self.deriv_coeff[deriv_string] == None:
            # Used pol_deg are in self._basis_dict[bootstrap][output_key][timestep]
            degrees = self._basis_dict["b_1"]
            derivative_degrees = copy.deepcopy(degrees)  # Power without deriavative
            derivative_coeffs = copy.deepcopy(
                self._coeffs_dict["b_1"]
            )  # Coefficient multipliers
            for key in list(degrees.keys()):
                for times in list(degrees[key].keys()):
                    derivative_coeffs[key][times] = np.ones(
                        (degrees[key][times].shape[0],)
                    )

            for par_name in deriv_names:
                par_num = self.par_names.index(par_name)
                for key in list(degrees.keys()):
                    for times in list(degrees[key].keys()):
                        for i in range(degrees[key][times].shape[0]):
                            deg = degrees[key][times][i]
                            # If the parameter is not there, set coefficient to 0
                            derivative_degrees[key][times][i][par_num] = 0
                            if deg[par_num] == 0:
                                derivative_coeffs[key][times][i] *= 0
                            else:
                                derivative_coeffs[key][times][i] *= 1

            # Assign the degrees correctly
            self.deriv_deg[deriv_string] = derivative_degrees
            self.deriv_coeff[deriv_string] = derivative_coeffs

    @decorator_deriv_check
    def derivative(self, samples, deriv_names, predict=True):
        """
        Calculates the derivative of the metamodel at the chosen points.
        The derivative is taken with respect to the set variable and to the chosen degree.
        Currently this function supports only degree == 1.
        Note: training with bootstrap not considered for this function!
        Note: this function returns only a mean, no stdev

        Parameters
        ----------
        samples : np.array
            Sample evaluations to get the derivative at.
        deriv_names : dict
            Description of the derivative. This includes name of the input
            parameters to use for the derivatives as dictionary keys, and
            the degree of derivative wrt the parameter as the value.
        predict : bool, optional
            The function returns the derivative psi-matrix if set to False, and the whole
            derivative prediction if set to True. The default is True.

        Returns
        -------
        mean_pred : dict
            Mean prediction for the derivative. Returned if predict is True.
        psi_dict : dict
            Psi-matrix for the derivative. Returned if predict is False.

        """
        # Get id of the parameter
        samples = self.input_space.transform(samples, method="user")
        deriv_string = json.dumps(deriv_names)

        #### Evaluate the derivative at the wanted points - return only mean, no std
        # Removed bootstrap from here, add again if needed
        n_max = np.max(self.pce_deg)
        poly_types = self.input_space.poly_types

        # The evaluation of the univariate basis have shape (#samples, #params, #degree)
        # Evaluate univariate basis for the non-derivative univariate polynomials
        univ_p_val_outer = eval_univ_basis(samples, n_max, poly_types, self._polycoeffs)
        pol_basis = self.deriv_deg[deriv_string]
        deriv_coeff = self.deriv_coeff[deriv_string]

        # Loop over outputs
        mean_pred = {}
        psi_dict = {}
        for output, values in pol_basis.items():
            mean = np.empty((len(samples), len(values)))
            psi_dict[output] = {}
            idx = 0
            for in_key, _ in values.items():
                basis_outer = pol_basis[output][in_key]
                psi_outer = create_psi(basis_outer, univ_p_val_outer)

                # Assemble Psi matrix for inner derivatives and degrees
                psi = psi_outer
                for par_name in deriv_names:
                    degree = deriv_names[par_name]

                    # Evaluate univariate basis for the derivative univariate polynomials
                    der_basis_coeffs = self.der_basis_coeffs[degree]
                    univ_p_val_inner = eval_univ_basis(
                        samples, n_max, poly_types, der_basis_coeffs
                    )

                    basis_inner = self.deriv_inner_deg[degree][par_name][output][in_key]
                    psi_inner = create_psi(basis_inner, univ_p_val_inner)
                    psi = np.multiply(psi, psi_inner)

                # Full Psi matrix obtained by multiplying the two parts
                coeff_mult = deriv_coeff[output][in_key]
                psi = np.multiply(psi, coeff_mult)
                psi_dict[output][in_key] = psi

                if predict:
                    # Prediction without error bar
                    coeffs = self._coeffs_dict["b_1"][output][in_key]
                    mean[:, idx] = np.dot(psi, coeffs)
                idx += 1

            # Save predictions for each output
            if predict:
                mean_pred[output] = mean

        if not predict:
            return psi_dict
        return mean_pred
