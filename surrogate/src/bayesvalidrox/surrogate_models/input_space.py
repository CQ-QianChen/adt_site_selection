#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input space built from set prior distributions
"""

import numpy as np
import chaospy
import scipy.stats as st
from .supplementary import check_ranges


# noinspection SpellCheckingInspection
class InputSpace:
    """
    This class generates the input space for the metamodel from the
    distributions provided using the `Input` object.

    Attributes
    ----------
    Input : obj
        Input object containing the parameter marginals, i.e. name,
        distribution type and distribution parameters or available raw data.
    meta_model_type : str
        Type of the meta_model_type.

    """

    def __init__(self, input_object, meta_model_type="pce"):
        self.input_object = input_object
        self.meta_model_type = meta_model_type

        # Other
        self.pce = None
        self.bound_tuples = None
        self.input_data_given = None
        self.j_dist = None
        self.mc_size = None
        self.ndim = None
        self.orig_j_dist = None
        self.par_names = None
        self.poly_types = None
        self.prior_space = None
        self.raw_data = None
        self.n_init_samples = None

        # Init
        self.check_valid_inputs()

    def check_valid_inputs(self) -> None:
        """
        Check if the given input_object is valid to use for further calculations:
        1) Has some Marginals
        2) The Marginals have valid priors
        3) All Marginals given as the same type (samples vs dist)

        Returns
        -------
        None

        """
        inputs = self.input_object
        self.ndim = len(inputs.marginals)

        # Check if PCE metamodel is selected.
        if self.meta_model_type.lower() == "pce":
            self.pce = True
        else:
            self.pce = False

        # check if marginals given
        if not self.ndim >= 1:
            raise AssertionError("Cannot build distributions if no marginals are given")

        # check that each marginal is valid
        for marginals in inputs.marginals:
            if len(marginals.input_data) == 0:
                if marginals.dist_type is None:
                    raise AssertionError("Not all marginals were provided priors")
            if np.array(marginals.input_data).shape[0] and (
                marginals.dist_type is not None
            ):
                raise AssertionError(
                    "Both samples and distribution type are given. Please choose only one."
                )

        # Check if input is given as dist or input_data.
        self.input_data_given = -1
        for marg in inputs.marginals:
            size = np.array(marg.input_data).shape[0]
            if size and abs(self.input_data_given) != 1:
                self.input_data_given = 2
                break
            if (not size) and self.input_data_given > 0:
                self.input_data_given = 2
                break
            if not size:
                self.input_data_given = 0
            if size:
                self.input_data_given = 1

        if self.input_data_given == 2:
            raise AssertionError(
                "Distributions cannot be built as the priors have different types"
            )

        # Get the bounds if input_data are directly defined by user:
        if self.input_data_given:
            for i in range(self.ndim):
                low_bound = np.min(inputs.marginals[i].input_data)
                up_bound = np.max(inputs.marginals[i].input_data)
                inputs.marginals[i].parameters = [low_bound, up_bound]

    # -------------------------------------------------------------------------
    def init_param_space(self, max_deg=1):
        """
        Initializes parameter space.

        Parameters
        ----------
        max_deg : int, optional
            Maximum degree. The default is `1`.

        Returns
        -------
        raw_data : array of shape (n_params, n_samples)
            Raw data.
        bound_tuples : list of tuples
            A list containing lower and upper bounds of parameters.

        """
        # Recheck all before running!
        self.check_valid_inputs()

        inputs = self.input_object
        ndim = self.ndim
        rosenblatt_flag = inputs.rosenblatt
        mc_size = 50000

        # Save parameter names
        self.par_names = []
        for par_idx in range(ndim):
            self.par_names.append(inputs.marginals[par_idx].name)

        # Create a multivariate probability distribution
        if max_deg is not None:
            j_dist, poly_types = self.build_polytypes(rosenblatt=rosenblatt_flag)
            self.j_dist, self.poly_types = j_dist, poly_types

        if self.input_data_given:
            self.mc_size = len(inputs.marginals[0].input_data)
            self.raw_data = np.zeros((ndim, self.mc_size))

            for par_idx in range(ndim):
                # Save parameter names
                try:
                    self.raw_data[par_idx] = np.array(
                        inputs.marginals[par_idx].input_data
                    )
                except:
                    self.raw_data[par_idx] = self.j_dist[par_idx].sample(mc_size)

        else:
            # Generate random samples based on parameter distributions
            self.raw_data = chaospy.generate_samples(mc_size, domain=self.j_dist)

        # Extract moments
        for par_idx in range(ndim):
            mu = np.mean(self.raw_data[par_idx])
            std = np.std(self.raw_data[par_idx])
            self.input_object.marginals[par_idx].moments = [mu, std]

        # Generate the bounds based on given inputs for marginals
        bound_tuples = []
        for i in range(ndim):
            if inputs.marginals[i].dist_type == "unif":
                low_bound = inputs.marginals[i].parameters[0]
                up_bound = inputs.marginals[i].parameters[1]
            else:
                low_bound = np.min(self.raw_data[i])
                up_bound = np.max(self.raw_data[i])

            bound_tuples.append((low_bound, up_bound))

        self.bound_tuples = tuple(bound_tuples)

    # -------------------------------------------------------------------------
    def build_polytypes(self, rosenblatt):
        """
        Creates the polynomial types to be passed to univ_basis_vals method of
        the MetaModel object.

        Parameters
        ----------
        rosenblatt : bool
            Rosenblatt transformation flag.

        Returns
        -------
        orig_space_dist : object
            A chaospy j_dist object or a gaussian_kde object.
        poly_types : list
            A list of polynomial types for the parameters.

        """
        inputs = self.input_object

        all_data = []
        all_dist_types = []
        orig_joints = []
        poly_types = []
        params = None

        for par_idx in range(self.ndim):

            if inputs.marginals[par_idx].dist_type is None:
                data = inputs.marginals[par_idx].input_data
                all_data.append(data)
                dist_type = None
            else:
                dist_type = inputs.marginals[par_idx].dist_type
                params = inputs.marginals[par_idx].parameters

            if rosenblatt:
                polytype = "hermite"
                dist = chaospy.Normal()

            elif dist_type is None:
                polytype = "arbitrary"
                dist = None

            elif "unif" in dist_type.lower():
                polytype = "legendre"
                if not np.array(params).shape[0] >= 2:
                    raise AssertionError("Distribution has too few parameters!")
                dist = chaospy.Uniform(lower=params[0], upper=params[1])
                
            # new: 
            elif "truncnorm" in dist_type.lower():
                polytype = "hermite"
                if not np.array(params).shape[0] >= 4:
                    raise AssertionError("Distribution has too few parameters!")
                a, b, loc, scale = params[0], params[1], params[2], params[3]
                dist = chaospy.TruncNormal(lower=a, upper=b, mu=loc, sigma=scale)

            elif "norm" in dist_type.lower() and "log" not in dist_type.lower():
                if not np.array(params).shape[0] >= 2:
                    raise AssertionError("Distribution has too few parameters!")
                polytype = "hermite"
                dist = chaospy.Normal(mu=params[0], sigma=params[1])

            elif "gamma" in dist_type.lower():
                polytype = "laguerre"
                if not np.array(params).shape[0] >= 3:
                    raise AssertionError("Distribution has too few parameters!")
                dist = chaospy.Gamma(shape=params[0], scale=params[1], shift=params[2])

            elif "beta" in dist_type.lower():
                if not np.array(params).shape[0] >= 4:
                    raise AssertionError("Distribution has too few parameters!")
                polytype = "jacobi"
                dist = chaospy.Beta(
                    alpha=params[0], beta=params[1], lower=params[2], upper=params[3]
                )

            elif "lognorm" in dist_type.lower():
                polytype = "hermite"
                if not np.array(params).shape[0] >= 2:
                    raise AssertionError("Distribution has too few parameters!")
                mu = np.log(params[0] ** 2 / np.sqrt(params[0] ** 2 + params[1] ** 2))
                sigma = np.sqrt(np.log(1 + params[1] ** 2 / params[0] ** 2))
                dist = chaospy.LogNormal(mu, sigma)
                # dist = chaospy.LogNormal(mu=params[0], sigma=params[1])

            elif "expon" in dist_type.lower():
                polytype = "exponential"
                if not np.array(params).shape[0] >= 2:
                    raise AssertionError("Distribution has too few parameters!")
                dist = chaospy.Exponential(scale=params[0], shift=params[1])

            elif "weibull" in dist_type.lower():
                polytype = "weibull"
                if not np.array(params).shape[0] >= 3:
                    raise AssertionError("Distribution has too few parameters!")
                dist = chaospy.Weibull(
                    shape=params[0], scale=params[1], shift=params[2]
                )

            else:
                message = (
                    f"DistType {dist_type} for parameter"
                    f"{par_idx + 1} is not available."
                )
                raise ValueError(message)

            if self.input_data_given or not self.pce:
                polytype = "arbitrary"

            # Store dists and poly_types
            orig_joints.append(dist)
            poly_types.append(polytype)
            all_dist_types.append(dist_type)

        # Prepare final output to return
        if None in all_dist_types:
            # Naive approach: Fit a gaussian kernel to the provided data
            data_ = np.asarray(all_data)
            try:
                orig_space_dist = st.gaussian_kde(data_)
            except Exception as exc:
                raise ValueError(
                    "The samples provided to the Marginals should be 1D only"
                ) from exc
            self.prior_space = orig_space_dist
        else:
            orig_space_dist = chaospy.J(*orig_joints)
            try:
                self.prior_space = st.gaussian_kde(orig_space_dist.sample(10000))
            except Exception as exc:
                raise ValueError(
                    "Parameter values are not valid, please set differently"
                ) from exc

        return orig_space_dist, poly_types

    # -------------------------------------------------------------------------
    def transform(self, X, params=None, method=None):
        """
        Transforms the samples via either a Rosenblatt or an isoprobabilistic
        transformation.

        Parameters
        ----------
        X : array of shape (n_samples,n_params)
            Samples to be transformed.
        params : list
            Parameters for laguerre/gamma-type distribution.
        method : string
            If transformation method is 'user' transform X, else just pass X.

        Returns
        -------
        tr_x: array of shape (n_samples,n_params)
            Transformed samples.

        """
        # Check for built j_dist
        if self.j_dist is None:
            raise AttributeError(
                "Call function init_param_space first to create j_dist"
            )

        # Check if X is 2d
        if X.ndim != 2:
            raise AttributeError("X should have two dimensions")

        # Check if size of X matches Marginals
        if X.shape[1] != self.ndim:
            raise AttributeError(
                "The second dimension of X should be the same size as the number of "
                "marginals in the input_object"
            )

        if self.input_object.rosenblatt:
            self.orig_j_dist, _ = self.build_polytypes(False)
            if method == "user":
                tr_x = self.j_dist.inv(self.orig_j_dist.fwd(X.T)).T
            else:
                # Inverse to original spcace -- generate sample ED
                tr_x = self.orig_j_dist.inv(self.j_dist.fwd(X.T)).T
        else:
            # Transform samples via an isoprobabilistic transformation
            _, n_params = X.shape
            inputs = self.input_object
            orig_j_dist = self.j_dist
            poly_types = self.poly_types

            disttypes = []
            for par_i in range(n_params):
                disttypes.append(inputs.marginals[par_i].dist_type)

            # Pass non-transformed X, if arbitrary PCE is selected.
            if None in disttypes or self.input_data_given or not self.pce:
                return X

            cdfx = np.zeros_like(X)
            tr_x = np.zeros_like(X)

            # Define the transformation function
            for par_i, (disttype, polytype) in enumerate(zip(disttypes, poly_types)):
                if disttype is not None:
                    dist = orig_j_dist[par_i]
                    cdf_func = dist.cdf
                else:
                    dist = None
                    cdf_func = None  # Identity function if disttype is None

                # Determine the transformation distribution and parameters
                if polytype == "legendre" or disttype == "uniform":
                    params_y = [-1, 1]
                    dist_y = st.uniform(
                        loc=params_y[0], scale=params_y[1] - params_y[0]
                    )
                elif polytype == "hermite" or disttype == "norm":
                    params_y = [0, 1]
                    dist_y = st.norm(loc=params_y[0], scale=params_y[1])
                elif polytype == "laguerre" or disttype == "gamma":
                    if params is None:
                        raise AttributeError(
                            "Additional parameters have to be set for the gamma distribution!"
                        )
                    params_y = [1, params[1]]
                    dist_y = st.gamma(a=params_y[0])
                else:
                    raise ValueError(
                        f"Unknown polytype or disttype: {polytype}, {disttype}"
                    )

                # Compute CDF and inverse CDF using vectorized operations
                if cdf_func is not None:
                    cdfx[:, par_i] = cdf_func(X[:, par_i])
                else:
                    cdfx[:, par_i] = X[:, par_i]
                tr_x[:, par_i] = dist_y.ppf(cdfx[:, par_i])

        return tr_x

    def random_sampler(self, n_samples, max_deg=1):
        """
        Samples the given raw data randomly.

        Parameters
        ----------
        n_samples : int
            Number of requested samples.

        max_deg : int, optional
            Maximum degree. The default is 1.
            This will be used to run init_param_space, if it has not been done
            until now.

        Returns
        -------
        samples: array of shape (n_samples, n_params)
            The sampling locations in the input space.

        """
        if self.raw_data is None:
            self.init_param_space(max_deg)
        else:
            if np.array(self.raw_data).ndim != 2:
                raise AttributeError(
                    "The given raw data for sampling should have two dimensions"
                )
        samples = np.zeros((n_samples, self.ndim))
        sample_size = self.raw_data.shape[1]

        # --- START OF CASE 1: Subsampling Existing Raw Data ---
        if n_samples < sample_size:
            # Draw ONE set of random indices for the full data set,
            # to keep potential parameter relationships
            rand_idx = np.random.choice(
                sample_size, n_samples, replace=False
            )  # Or replace=True if allowed
            # Select the samples using those indices
            # Assuming raw_data shape is (n_dim, n_samples), you transpose/reorder
            samples = self.raw_data[:, rand_idx].T

        # --- START OF CASE 2: Sampling from Distribution (n_samples >= sample_size) ---
        else:
            if self.j_dist is None:
                raise AttributeError(
                    "Sampling cannot proceed, build InputSpace "
                    "with max_deg != 0 to create j_dist!"
                )
            try:
                # Use resample if j_dist is of type gaussian_kde
                samples = self.j_dist.resample(int(n_samples)).T
            except AttributeError:
                # Use sample if j_dist is of type chaospy.J
                samples = self.j_dist.sample(int(n_samples)).T

            # If there is only one input transform the samples
            if samples.ndim == 1:
                samples = np.swapaxes(np.atleast_2d(samples), 0, 1)
            else:
                if samples.shape[0] != n_samples:
                    samples = samples.T

            # --- EFFICIENT BATCH BOUNDARY CHECKING AND REJECTION SAMPLING ---
            max_iter = 100
            iteration = 0

            # 1. Initialize the mask for invalid samples
            # A sample is invalid if check_ranges returns False
            is_invalid = np.array(
                [
                    not check_ranges(param_set, self.bound_tuples)
                    for param_set in samples
                ]
            )
            while np.any(is_invalid) and iteration < max_iter:

                n_invalid = np.sum(is_invalid)

                # 2. Generate new replacement samples for ALL invalid slots
                # This logic mimics the original preference for chaospy.generate_samples
                # over resample/sample if possible.
                try:
                    new_samples = chaospy.generate_samples(
                        n_invalid, domain=self.j_dist, rule="random"
                    ).T
                except:
                    # Fallback to the method used for initial sampling
                    try:
                        new_samples = self.j_dist.resample(n_invalid).T
                    except AttributeError:
                        new_samples = self.j_dist.sample(n_invalid).T

                # Ensure shape is correct for ndim=1 case
                if new_samples.ndim == 1:
                    new_samples = np.swapaxes(np.atleast_2d(new_samples), 0, 1)
                else:
                    if new_samples.shape[0] != n_invalid:
                        new_samples = new_samples.T

                # 3. Replace the invalid samples in the original array using the mask
                samples[is_invalid] = new_samples

                # 4. Re-evaluate the validity of ALL samples
                is_invalid = np.array(
                    [
                        not check_ranges(param_set, self.bound_tuples)
                        for param_set in samples
                    ]
                )

                iteration += 1

            # 5. Safety check for infinite loop / highly constrained distributions
            if iteration == max_iter and np.any(is_invalid):
                raise RuntimeError(
                    f"Failed to find valid samples within bounds after {max_iter} iterations. "
                    "The distribution might be too wide or the bounds too restrictive."
                )

        return samples
