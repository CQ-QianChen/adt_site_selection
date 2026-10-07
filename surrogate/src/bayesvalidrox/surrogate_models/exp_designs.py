#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimental design with associated sampling methods
"""

import itertools
import math
import warnings

import chaospy
import h5py
import numpy as np

from .apoly_construction import apoly_construction
from .input_space import InputSpace


class ExpDesigns(InputSpace):
    """
    This class generates samples from the prescribed marginals for the model
    parameters using the `Input` object.

    Attributes
    ----------
    input_object : obj
        Input object containing the parameter marginals, i.e. name,
        distribution type and distribution parameters or available raw data.
    meta_model_type : str
        Type of the meta_model_type.
    sampling_method : str
        Name of the sampling method for the experimental design. The following
        sampling method are supported:

        * random
        * latin_hypercube
        * sobol
        * halton
        * hammersley
        * chebyshev(FT)
        * grid(FT)
        * pcm
        * trunc_pcm
        * user
    hdf5_file : str
        Name of the hdf5 file that contains the experimental design.
    n_new_samples : int
        Number of (initial) training points.
    n_max_samples : int
        Number of maximum training points.
    tradeoff_scheme : str
        Trade-off scheme to assign weights to the exploration and exploitation
        scores in the sequential design. The following methods are supported:
        * Explore_only
        * Exploit_only
        * Equal
        * Epsilon-decreasing
        * Adaptive
    n_candidates : int
        Number of candidate training sets to calculate the scores for.
    explore_method : str
        Type of the exploration method for the sequential design. The following
        methods are supported:
        * Voronoi
        * random
        * latin_hypercube
        * LOOCV
        * dual annealing
    exploit_method : str
        Type of the exploitation method for the sequential design. The
        following methods are supported:
        * BayesActDesign
        * VarOptDesign
        * Alphabetic
    util_func : str
        The utility function to be specified for the `exploit_method`. For the
        available utility functions see Note section.
    n_cand_groups : int
        Number of candidate groups. Each group of candidate training sets will
        be evaulated separately in parallel.

    Note
    ----------
    The following utiliy functions for the **exploitation** methods are
    supported:

    #### BayesActDesign -> when data is available
    - DKL
    - BME
    - IE
    - DIC
    - Bayesrisk
    - DPP (D-Posterior-percision)
    - APP (A-Posterior-percision)

    #### VarBasedOptDesign -> when data is not available
    - ALM (Entropy/MMSE/active learning MacKay)
    - EIGF (Expected Improvement for Global fit)
    - MI (Mutual information)
    - ALC (Active learning Cohn)

    #### alphabetic
    - D-Opt (D-Optimality)
    - A-Opt (A-Optimality)
    - K-Opt (K-Optimality)
    """

    def __init__(
        self,
        input_object,
        meta_model_type="pce",
        sampling_method="random",
        hdf5_file=None,
        n_init_samples=1,
        n_new_samples=1,
        n_max_samples=None,
        tradeoff_scheme="explore_only",
        n_candidates=1000,
        explore_method="random",
        exploit_method="varoptdesign",
        util_func="ALM",
        n_cand_groups=4,
        max_func_itr=1,
        out_dir="",
    ):

        super().__init__(input_object, meta_model_type)
        # Sampling settings
        self.input_object = input_object
        self.meta_model_type = meta_model_type
        self.sampling_method = sampling_method
        self.hdf5_file = hdf5_file
        self.out_dir = out_dir

        # Training settings
        self.n_init_samples = n_init_samples
        self.n_new_samples = n_new_samples
        self.n_max_samples = (
            n_max_samples if n_max_samples is not None else n_init_samples
        )
        self.explore_method = explore_method
        self.exploit_method = exploit_method
        self.util_func = util_func
        self.tradeoff_scheme = tradeoff_scheme
        self.n_candidates = n_candidates
        self.n_cand_groups = n_cand_groups
        self.max_func_itr = max_func_itr  # for the seqDesign class

        # Other
        self.ndim = None
        self.x_values = None
        self.x = None
        self.y = None
        self.out_names = None
        self.x_valid = None
        self.y_valid = None

        # Init
        self.check_valid_inputs()
        if self.out_dir == "":
            self.out_dir = "Outputs_Priors/"

    # -------------------------------------------------------------------------

    def generate_samples(self, n_samples, sampling_method="random"):
        """
        Generates samples with given sampling method

        Parameters
        ----------
        n_samples : int
            Number of requested samples.
        sampling_method : str, optional
            Sampling method. The default is `'random'`.

        Returns
        -------
        samples: array of shape (n_samples, n_params)
            Generated samples from defined model input object.

        """
        if not self.input_data_given:
            # If sampling method is 'grid', adjust n_samples so that the total number
            # of samples matches the requested n_samples (n_samples**n_dim)
            if sampling_method.lower() == "grid":
                n_samples = int(np.ceil(n_samples ** (1 / self.ndim)))

            if self.raw_data is None:
                self.init_param_space()

            # Case I = User passed known distributions
            try:
                samples = chaospy.generate_samples(
                    int(n_samples), domain=self.j_dist, rule=sampling_method
                )
            except Exception:
                # Fail fast with a clear error if chaospy sampling fails
                raise RuntimeError(
                    f"Sampling using chaospy with rule '{sampling_method}' failed."
                    " Select a valid method"
                )
        else:
            # Case II: Input values are directly given by the user.
            if sampling_method != "random":
                warnings.warn(
                    "Defaulting to random sampling as prior samples are provided."
                )
            samples = self.random_sampler(int(n_samples)).T

        return samples.T

    # -------------------------------------------------------------------------
    def generate_ed(self, max_deg=1):
        """
        Generates experimental designs (training set) with the given method.

        Parameters
        ----------
        max_deg : int, optional
            Maximum (PCE) polynomial degree. The default is 1.

        Returns
        -------
        None

        """
        # Read ExpDesign (training and targets) from the provided hdf5
        if self.hdf5_file is not None:
            if self.out_names is None:
                raise AttributeError(
                    "ExpDesign cannot be read without valid out_names."
                )
            self.read_from_file(self.out_names)

        # Generate the samples based on requested method
        self.init_param_space(max_deg)

        # Case 1: X is given -> sampling_method = 'user'
        if self.x is not None:
            self.x = np.array(self.x)
            if self.sampling_method != "user":
                self.sampling_method = "user"
                warnings.warn(
                    "The sampling method has been switched to 'user' and "
                    "the given ExpDesign.x is used."
                )
            if self.x.ndim != 2:
                raise AttributeError("The provided samples should have 2 dimensions")
            if self.n_init_samples is None:
                self.n_init_samples = self.x.shape[0]
            self.n_init_samples = len(self.x)
            return

        # Error if 'user' and no samples given
        if self.sampling_method == "user":
            raise AttributeError(
                "User-defined sampling cannot proceed as "
                "no samples provided. Please add them to "
                "this class as attribute X"
            )

        # Warning that outputs will be rerun if X is sampled anew
        if self.y is not None:
            self.y = None
            warnings.warn(
                "The given model outputs will be overwritten for the chosen ExpDesign settings."
            )

        # Case 2 : X is not given
        n_samples = int(self.n_init_samples)
        if n_samples < 0:
            raise ValueError(
                "A negative number of samples cannot be created. "
                "Please provide positive n_samples"
            )

        # Sample the distribution of parameters
        samples = None
        if self.sampling_method.lower() in ["pcm", "trunc_pcm"]:
            samples = self.pcm_sampler(n_samples, max_deg)
        else:
            samples = self.generate_samples(n_samples, self.sampling_method)

        self.x = samples
        self.n_init_samples = samples.shape[0]  # Update n_init_samples, in case of grid

    def read_from_file(self, out_names):
        """
        Reads in the ExpDesign from a provided h5py file and saves the results.

        Parameters
        ----------
        out_names : list of strings
            The keys that are in the outputs (y) saved in the provided file.

        Returns
        -------
        None.

        """
        if self.hdf5_file is None:
            raise AttributeError(
                "ExpDesign cannot be read in, please provide hdf5 file first"
            )

        # Read hdf5 file
        with h5py.File(self.hdf5_file, "r+") as f:

            # Read EDX and pass it to ExpDesign object
            try:
                self.x = np.array(f["EDX/New_init_"])
            except KeyError:
                self.x = np.array(f["EDX/init_"])

            # Update number of initial samples
            self.n_init_samples = self.x.shape[0]

            # Read EDX and pass it to ExpDesign object
            self.y = {}

            # Extract x values
            try:
                self.y["x_values"] = {}
                for _, var in enumerate(out_names):
                    x = np.array(f[f"x_values/{var}"])
                    self.y["x_values"][var] = x
            except KeyError:
                self.y["x_values"] = np.array(f["x_values"])

            # Store the output
            for _, var in enumerate(out_names):
                try:
                    y = np.array(f[f"EDY/{var}/New_init_"])
                except KeyError:
                    y = np.array(f[f"EDY/{var}/init_"])
                self.y[var] = y
        f.close()
        print(f"Experimental Design is read in from file {self.hdf5_file}")
        print("")

    # -------------------------------------------------------------------------
    def pcm_sampler(self, n_samples, max_deg):
        """
        Generates collocation points based on the root of the polynomial degrees
        (point collocation method).
        PCM options:
        - pcm: all collocation points up to a max polynomial degree,
                regardless of n_samples
        - trunc_pcm: truncated collocation points
                (only a subset, based on user-given number of samples)

        Parameters
        ----------
        n_samples : int
            Number of requested  by user.
        max_deg : int
            Maximum degree defined by user. Will also be used to run
            init_param_space if that has not been done beforehand.

        Returns
        -------
        opt_col_points: array of shape (n_samples, n_params)
            Collocation points.

        """

        if self.raw_data is None:
            self.init_param_space(max_deg)

        raw_data = self.raw_data

        def num_collocation_points(deg, ndim):
            """Number of points in a full tensor grid for `deg` and `ndim`."""
            return (deg + 1) ** ndim

        if self.sampling_method == "pcm":
            guess_deg = max_deg
            warnings.warn(
                f"Using max_deg={max_deg} to estimate quadrature collocation points. "
                f"User-set `n_samples` will be overwritten"
            )
        else:  # trunc_pcm
            # Pick the smallest degree that yields at least n_samples points
            for d in range(1, max_deg + 1):
                if num_collocation_points(d, self.ndim) >= n_samples:
                    guess_deg = d
                    break
            else:
                # If even max_deg cannot give enough points, fallback to max_deg
                guess_deg = max_deg

        # ------------------------------------------------
        # Construction of collocation points per parameter
        c_points = np.zeros((guess_deg + 1, self.ndim))

        def polynomial_pa(par_idx):
            """Constructs the polynomial expansion for a given parameter index."""
            return apoly_construction(self.raw_data[par_idx], max_deg)

        for i in range(self.ndim):
            poly_coeffs = polynomial_pa(i)[guess_deg + 1][::-1]
            c_points[:, i] = np.trim_zeros(np.roots(poly_coeffs))

        #  Construction of optimal integration points
        prod = itertools.product(np.arange(1, guess_deg + 2), repeat=self.ndim)
        sort_dig_unique_combos = np.array(list(filter(lambda x: x, prod)))

        # ------------------------------------------------
        # Rank points based on a given criteria
        def rank_points(c_points, raw_data, criterion="mean_distance"):
            temp_ = None
            # Ranking relatively mean distance from existing data points
            if criterion == "mean_distance":
                temp_ = np.abs(c_points - np.mean(raw_data, axis=1, keepdims=True).T)
            elif criterion == "magnitude":
                # Ranking based on magnitude of collocation points
                temp_ = np.abs(c_points)

            # Add more criteria if needed (e.g., variance-based)
            return np.argsort(temp_, axis=0)

        index_cp = rank_points(c_points, raw_data)

        # -------------------------------------------------
        # Sorted collocation points based on the ranking
        sort_cpoints = np.empty((0, guess_deg + 1))

        for j in range(self.ndim):
            sort_cp = c_points[index_cp[:, j], j]
            sort_cpoints = np.vstack((sort_cpoints, sort_cp))

        # Mapping of Combination to Cpoint Combination
        sort_unique_combos = np.empty(shape=[0, self.ndim])
        for i in range(len(sort_dig_unique_combos)):
            sort_un_comb = []
            sort_uni_comb = None
            for j in range(self.ndim):
                sort_uc = sort_cpoints[j, sort_dig_unique_combos[i, j] - 1]
                sort_un_comb.append(sort_uc)
                sort_uni_comb = np.asarray(sort_un_comb)
            sort_unique_combos = np.vstack((sort_unique_combos, sort_uni_comb))

        # Output the collocation points
        if self.sampling_method.lower() == "trunc_pcm":
            # trunc-pcm
            opt_col_points = sort_unique_combos[0:n_samples]
            if opt_col_points.shape[0] < n_samples:
                warnings.warn(
                    f"Generated collocation points {opt_col_points.shape[0]}"
                    f" are less than requested n_samples={n_samples}. "
                    f"Consider increasing max_deg to generate more collocation points."
                )
        else:
            # full-pcm
            opt_col_points = sort_unique_combos

        # Update number of initial samples
        self.n_init_samples = opt_col_points.shape[0]

        return opt_col_points
