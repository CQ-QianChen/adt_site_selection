#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parent-class for GSA.
"""

import os
import json
import copy
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))
DPI = 100


class GSA:
    """
    This class provides the template for global sensitivity analysis classes.

    Parameters
    ----------
    engine : obj
        Trained Engine object, is expected to contain a trained MetaModel object.
    name : string
        Name of the object to be used for saving the generated files.
        The default is 'model'.
    out_dir : string
        Output directory in which the GSA results are placed.
        The results are contained in a subfolder '/Outputs_GSA_name'
        The default is ''.
    out_format : string
        Format of the saved plots. Supports 'png' and 'pdf'. The default is 'pdf'.
    n_mc : int, optional
        The number of MC samples for estimating moments. Only used if metamodel is not
        given, or not of PCE-type.
        The default is 1000.
    n_bins : int, optional
        The number of bins for estimating moments. Only used if metamodel is not
        given, or not of PCE-type.
        The default is 100.

    Raises
    ------
    AttributeError
        `engine` must be trained.

    """

    def __init__(
        self, engine, name="model", out_dir="", out_format="pdf", n_mc=1000, n_bins=10
    ):
        self.engine = engine
        self.name = name
        self.out_format = out_format
        self.n_mc = n_mc
        self.n_bins = n_bins

        # Initialize attributes
        self.plot_type = ""
        self.xlabel = "Time [s]"
        self.out_dir = f"./{out_dir}/Outputs_GSA/"
        if self.name != "":
            self.out_dir = f"./{out_dir}/Outputs_GSA_{self.name}/"

        # Engine settings
        self.par_names = []
        self.x_values = np.zeros(1)

        # GSA-related
        self.moment_functions = {
            "mean": np.mean,
            "var": np.var,
            "skew": stats.skew,
            "kurt": stats.kurtosis,
        }

        self.setup()

    def setup(self):
        """
        Set up and check the inits for the class.
        """
        # GSA only available for trained engines
        if not self.engine.trained:
            raise AttributeError("GSA can only be performed on trained engines.")

        # Settings from the engine
        self.par_names = self.engine.exp_design.par_names
        self.x_values = self.engine.exp_design.x_values
        if not isinstance(self.x_values, np.ndarray):
            raise TypeError(
                "Engine x_values must be either a numpy array"
                " or a dict with output keys, got {type(self.x_values)}."
            )

        # Create output folder for the plots
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def check_weights(self, engines, model_weights):
        """
        Check the given model-weights for correctness.

        Parameters
        ----------
        engines : dict
            Dictionary of engines that the weights belong to.
        model_weights : dict
            Dictionary of model weights for the engines.
            Should have the same keys as the engines and sum to 1.

        Returns
        -------
        model_weights : dict
            Fixed model weights.
        """
        # Check model-weights
        if model_weights is None and engines is not None:
            print("No weights given, assuming equal weights.")
            n_engines = len(list(engines.keys()))
            model_weights = {e_name: 1 / n_engines for e_name in engines}

        if not isinstance(model_weights, dict):
            raise TypeError("The model weights should be given as a dict.")
        if engines is not None:
            for e_name in engines:
                if e_name not in model_weights:
                    raise AttributeError(f"No weight given for engine {e_name}.")
        weights = np.array(list(model_weights.values()))
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("The model weights should be between 0 and 1.")
        if np.sum(weights) != 1:
            raise ValueError("The model weights should add to 1.")
        return model_weights

    def par_to_parid(self, parameters, engine=None):
        """
        Transforms list of parameter names into corresponding list of parameter ids.

        Parameters
        ----------
        parameters : list
            Parameters to get the IDs for.
        engine : obj, optional
            Trained BVR engine. If this is given,
            the parameter IDs are obtained from this engine,
            otherwise self.engine is used.
            The default is None.

        Returns
        -------
        par_id : list
            List of parameter IDs.

        """
        if engine is None:
            engine = self.engine

        # Check parameter
        if not isinstance(parameters, list):
            raise TypeError("The parameters should be given as a list.")

        par_id = []
        for parameter in parameters:
            par_names = engine.exp_design.par_names
            if parameter not in par_names:
                raise AttributeError(
                    f"The set parameter {parameter} not found in the inputspace."
                )
            par_id.append(par_names.index(parameter))
        return par_id

    def parid_to_notparid(self, par_ids, engine=None):
        """
        Get the list of all parameter-IDs that are not in par_id.

        Parameters
        ----------
        par_ids : list
            List of paramter IDs.
        engine : obj, optional
            Trained BVR engine. If this is given,
            the parameter IDs are obtained from this engine,
            otherwise self.engine is used.
            The default is None.

        Returns
        -------
        not_par_id : list
            List of all other parameter IDs.

        """
        if engine is None:
            engine = self.engine

        # Check parameter
        if not isinstance(par_ids, list):
            raise TypeError("The parameters should be given as a list.")

        not_par_id = []
        for par in range(len(engine.exp_design.par_names)):
            if par not in par_ids:
                not_par_id.append(par)
        return not_par_id

    def generate_mcsamples(self, engines, n_mc=1000):
        """
        Generate and store mc-samples from a set of models.

        Parameters
        ----------
        engines : dict
            Engines to generate the samples from.
        n_mc : int, optional
            Number of mc samples to generate.
            The default is 1000.

        Returns
        -------
        mc_samples, mc_out : dict, dict
            Samples generated from each engine.
        """
        # Check inputs
        if not isinstance(engines, dict):
            raise TypeError("The engines should be given as a dictionary.")

        if not isinstance(n_mc, int):
            raise TypeError("The number of MC-samples has to be an integer.")
        if n_mc <= 0:
            raise ValueError("Only positive numbers of MC-samples allowed.")

        mc_samples = {}
        mc_outputs = {}
        for name, engine in engines.items():
            out, _, samples = engine.eval_metamodel(nsamples=n_mc, return_samples=True)
            mc_samples[name] = samples
            mc_outputs[name] = out
        return mc_samples, mc_outputs

    def calc_model_moment(self, data):
        """
        Calculate the statistical moments conditional on the model choice.
        The statistics are calculated for every given moment, engine and output.

        Parameters
        ----------
        data : dict
            MC outputs across with keys matching the given engines.

        Returns
        -------
        model_moment: dict
            Dictionary of the moments for each model.
            Stacking of keys is:
                moment - engine - out_key
        """
        if not isinstance(data, dict):
            raise TypeError("The MC-data should be given as a dict.")

        model_moment = {}
        for m_id, moment in self.moment_functions.items():
            model_moment[m_id] = {}
            for name, out in data.items():
                stat = {}
                for key in out:
                    if key == "x_values":
                        continue
                    stat[key] = np.zeros(out[key].shape[1])
                    for t in range(out[key].shape[1]):
                        stat[key][t] = moment(out[key][:, t])
                    stat[key] = np.array(stat[key])
                model_moment[m_id][name] = stat
        return model_moment

    def calc_full_moment(self, model_moments, model_weights):
        """
        Calculate the metrics across the available models.
        Returns metrics for
         - mean
         - var
         - skewness
         - kurtosis

        Parameters
        ----------
        model_moments : dict
            Moments of the group of models, that should be combined.
        model_weights : dict
            Weights for each available model.

        Returns
        -------
        full_stats : dict
            Dictionary with grouped moments.
        """
        # Check inputs
        if not isinstance(model_moments, dict):
            raise TypeError("Model-specific moments should be given as dict.")
        for moment in self.moment_functions:
            if moment not in model_moments:
                raise AttributeError(f"Model-specific moments for {moment} not given.")
        model_weights = self.check_weights(None, model_weights)

        # Init returns
        full_stats = {}
        t_len = self.x_values.shape[0]
        e_name = list(model_weights.keys())[0]

        # Mean can be calculated directly from the mc-samples
        full_stats["mean"] = {}
        for key in self.engine.out_names:
            full_stats["mean"][key] = np.zeros(model_moments["mean"][e_name][key].shape)
            for t in range(t_len):
                wm_term = 0
                for name, weight in model_weights.items():
                    # Only within-model term
                    wm_term += weight * model_moments["mean"][name][key][t]
                full_stats["mean"][key][t] = wm_term

        # Calculate the variance from the mean
        full_stats["var"] = {}
        for key in self.engine.out_names:
            full_stats["var"][key] = np.zeros(model_moments["mean"][e_name][key].shape)
            for t in range(t_len):
                wm_term, bm_term = 0, 0
                for name, weight in model_weights.items():
                    # Within-model term
                    wm_term += weight * model_moments["var"][name][key][t]

                    # Between-model term
                    bm_term += weight * np.square(
                        full_stats["mean"][key][t] - model_moments["mean"][name][key][t]
                    )

                full_stats["var"][key][t] = wm_term + bm_term

        # Calculate the skewness
        full_stats["skew"] = {}
        for key in self.engine.out_names:
            full_stats["skew"][key] = np.zeros(model_moments["mean"][e_name][key].shape)
            for t in range(t_len):
                wm_term, bm_term, mx_term = 0, 0, 0
                for name, weight in model_weights.items():
                    mean_diff = (
                        full_stats["mean"][key][t] - model_moments["mean"][name][key][t]
                    )
                    # Within-model term
                    full_var = full_stats["var"][key][t]
                    model_var = model_moments["var"][name][key][t]
                    wm_term += (
                        weight
                        * np.power(model_var / full_var, 3 / 2)
                        * model_moments["skew"][name][key][t]
                    )

                    # Between-model term
                    bm_term += (
                        weight * np.power(mean_diff, 3) / np.power(full_var, 3 / 2)
                    )

                    # Mixed term
                    mx_term += (
                        weight * model_var / np.power(full_var, 3 / 2) * mean_diff
                    )

                full_stats["skew"][key][t] = wm_term - bm_term - 3 * mx_term

        # Calculate the kurtosis
        full_stats["kurt"] = {}
        for key in self.engine.out_names:
            full_stats["kurt"][key] = np.zeros(model_moments["mean"][e_name][key].shape)
            for t in range(t_len):
                wm_term, bm_term, mx_term = 0, 0, 0
                for name, weight in model_weights.items():
                    mean_diff = (
                        full_stats["mean"][key][t] - model_moments["mean"][name][key][t]
                    )
                    full_var = full_stats["var"][key][t]
                    model_var = model_moments["var"][name][key][t]
                    model_skew = model_moments["skew"][name][key][t]

                    # Within-model
                    wm_term += (
                        weight
                        * model_moments["kurt"][name][key][t]
                        * np.power(model_var / full_var, 2)
                    )
                    bm_term += weight * np.power(mean_diff, 4) / np.power(full_var, 2)
                    mx_term += (
                        6
                        * weight
                        * model_skew
                        * np.power(model_var, 5 / 2)
                        / np.power(full_var, 2)
                        * mean_diff
                    )

                full_stats["kurt"][key][t] = wm_term + bm_term + mx_term

        return full_stats

    def calc_par_moment(self, samples, data, par_ids, n_bins, return_bins=False):
        """
        Calculate the moments jointly conditional on the given paramters
        using binning.

        The code is inspired by the GSA.UN package for R.
        Link to specific file: https://github.com/cagarciae/GSA.UN/blob/master/R/Cond_Moments.R

        Parameters
        ----------
        samples : np.array
            MC samples
        data : dict
            Model outputs on MC samples
        par_inds : list
            Ids of the parameters to condition on
        n_bins : int
            Number of bins to use.
        return_bins : bool, optional
            Set to True to return the mean sample value in each bin.
            The default is False.

        Returns
        -------
        par_moments : dict
            Output moments conditional on the given parameters.
        bins : np.ndarray
            Average parameter values for each selected bin.
            Only returned if return_bins=True.

        """
        # Check inputs
        if not isinstance(samples, np.ndarray):
            raise TypeError("The samples should be given as an array.")
        if not isinstance(data, dict):
            raise TypeError("The data should be given as a dictionary.")
        if not isinstance(par_ids, list):
            raise TypeError("The parameter ids should be given as a list.")
        if not isinstance(n_bins, int):
            raise TypeError("The number of bins should be an integer.")
        if n_bins <= 0:
            raise ValueError("The number of bins should be positive and nonzero.")

        # Set n_bins as the number of bins along each param dimension
        n_par = len(par_ids)
        n_axis_bins = int(np.power(n_bins, 1 / n_par))
        if n_par > 1:
            print(
                f"{n_par} axes found, adjusted axis-wise bin to {n_axis_bins} ({n_bins} overall)."
            )

        # Calculate the ranges for each parameter
        bin_size = int(samples.shape[0] / n_axis_bins)
        ranges = np.zeros((len(par_ids), n_axis_bins, 3))
        axis_bin_ids = []
        for p_id, par in enumerate(par_ids):
            axis_bin_ids.append(np.arange(0, n_axis_bins, 1))
            sorted_samples = np.sort(samples[:, par])
            for b_id in range(n_axis_bins):
                bin_samples = sorted_samples[b_id * bin_size : (b_id + 1) * bin_size]
                ranges[p_id, b_id, 0] = np.min(bin_samples)
                ranges[p_id, b_id, 1] = np.mean(bin_samples)
                ranges[p_id, b_id, 2] = np.max(bin_samples)

        # Get the samples in each bin
        bin_ids = {}
        for b_id in range(n_bins):
            bin_ids[b_id] = []
        if n_par == 1:
            bin_combinations = np.array(axis_bin_ids)

            for b_id in range(n_bins):
                for s_id, s in enumerate(samples):
                    for p_id, par in enumerate(par_ids):
                        if (
                            s[par] >= ranges[p_id, b_id, 0]
                            and s[par] <= ranges[p_id, b_id, 2]
                        ):
                            bin_ids[b_id].append(s_id)
        else:
            # Get all the combinations - expensive for higher dim
            bin_combinations = np.array(list(itertools.product(*axis_bin_ids)))

            for s_id, s in enumerate(samples):
                bin_options = np.ones(bin_combinations.shape[0])  # []
                for p_id, par in enumerate(par_ids):
                    for b_id in range(n_axis_bins):
                        if (
                            s[par] >= ranges[p_id, b_id, 0]
                            and s[par] <= ranges[p_id, b_id, 2]
                        ):
                            bin_options *= bin_combinations[:, p_id] == b_id
                            continue
                bin_ = np.where(bin_options == 1)[0][0]
                bin_ids[bin_].append(s_id)

        # Remove bins without samples
        for bin_ in range(n_bins):
            if len(bin_ids[bin_]) == 0:
                bin_ids.pop(bin_)
        n_bins_adjust = len(list(bin_ids.keys()))

        # Iterate through the outputs
        par_moment = {}
        bins = np.zeros((n_bins_adjust))
        for out_key in data:
            if out_key == "x_values":
                continue

            # Setup of return
            d_len = data[out_key].shape[1]
            for moment in self.moment_functions:
                if moment not in par_moment:
                    par_moment[moment] = {}
                par_moment[moment][out_key] = np.zeros((n_bins, d_len))

            # Iterate through time, bins
            for t_step in range(data[out_key].shape[1]):
                for b_id, bin_ in enumerate(list(bin_ids.keys())):
                    # Sort data matching the samples
                    bin_data = data[out_key][bin_ids[bin_], t_step]

                    # Calculate the metrics
                    for moment, m_func in self.moment_functions.items():
                        par_moment[moment][out_key][b_id, t_step] = m_func(bin_data)
                    bins[b_id] = np.mean(samples[bin_ids[bin_], par_ids[-1]])

        if return_bins:
            return par_moment, bins
        return par_moment

    def store_to_json(self, indices):
        """
        Store indices to json file.

        Parameters
        ----------
        indices : dict
            Dictionary of all indices to store, the outermost keys
            should be the index names.

        """
        store_copy = copy.deepcopy(indices)
        for idx_key, _ in store_copy.items():
            # Trafo ndarrays to lists
            for key in store_copy[idx_key]:
                for out_key in self.engine.out_names:
                    store_copy[idx_key][key][out_key] = store_copy[idx_key][key][
                        out_key
                    ].tolist()

        filename = f"{self.out_dir}/indices.json"
        with open(filename, "w") as file:
            json.dump(store_copy, file)

    def plot(self, indices: dict, plot_type: str = "line"):
        """
        Plot sensitivity indices.

        Parameters
        ----------
        indices : dict
            Dictionary of indices.
        plot_type: str, optional
            Type of plot, supports 'line' and 'bar'.
            The default is 'line'.

        """
        # Check settings
        if plot_type not in ["line", "bar"]:
            raise AttributeError("The wanted plot type is not supported.")

        for ind_name, index in indices.items():
            for out_key in self.engine.out_names:

                # Start plotting
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])

                if plot_type == "bar":
                    df_data = {self.xlabel: self.x_values}

                    # Generate df to plot - use only mean
                    for par, values in index.items():
                        df_data[par] = np.mean(values[out_key], axis=0)
                    # Generate error bars - stdev
                    err = copy.deepcopy(df_data)
                    for par, values in index.items():
                        err[par] = np.std(values[out_key], axis=0)

                    df_data = pd.DataFrame(df_data)
                    df_data.plot(
                        x=self.xlabel,
                        y=list(index.keys()),
                        kind="bar",
                        ax=ax,
                        rot=0,
                        colormap="Dark2",
                        yerr=err,
                    )
                else:  # line plot
                    for par, values in index.items():
                        means = values[out_key]
                        stdevs = np.zeros(means.shape)
                        if len(means.shape) > 1:
                            means = np.mean(means, axis=0)
                            stdevs = np.std(means, axis=0)
                        ax.plot(
                            self.x_values,
                            means,
                            label=par,
                            marker="x",
                            lw=2.5,
                        )
                        ax.fill_between(
                            self.x_values,
                            means - stdevs,
                            means + stdevs,
                            alpha=0.15,
                            color="gray",
                        )
                    ax.set_ylim(bottom=0)

                # Labels and titles
                ax.set_ylabel(ind_name)
                ax.set_xlabel(self.xlabel)
                ax.legend(loc="best", frameon=True)
                ax.set_title(f"{ind_name} of {out_key}")

                # Save plot
                fig.savefig(
                    f"{self.out_dir}/{ind_name}_{out_key}.{self.out_format}",
                    bbox_inches="tight",
                )
                plt.close(fig)
