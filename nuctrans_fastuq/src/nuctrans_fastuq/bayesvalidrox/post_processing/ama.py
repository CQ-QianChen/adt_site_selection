#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of postprocessing functions into a class.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .gsa import GSA


class AMA(GSA):
    """
    This class provides functions for global sensitivity analysis of trained engines,
    specifically for the AMA-indices.

    Parameters
    ----------
    engine : obj
        Trained Engine object, is expected to contain a trained MetaModel object.
    name : string
        Name of the PostProcessing object to be used for saving the generated files.
        The default is 'model'.
    out_dir : string
        Output directory in which the PostProcessing results are placed.
        The results are contained in a subfolder '/Outputs_PostProcessing_name'
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
        self, engine, name="model", out_dir="", out_format="pdf", n_mc=1000, n_bins=100
    ):
        # Use parent init
        super().__init__(
            engine,
            name=name,
            out_dir=out_dir,
            out_format=out_format,
            n_mc=n_mc,
            n_bins=n_bins,
        )

        self.scale_diff = 100
        self.index_names = ["AMAE", "AMAV", "AMAs", "AMAk"]

    def setup(self):
        """
        Extend GSA setup for AMA.
        """
        super().setup()

        # Adapt output directory
        if self.out_dir[:2] != "./":
            self.out_dir = f"./{self.out_dir}"

        if self.out_dir[-4:] != "/ama":
            self.out_dir = f"{self.out_dir}/ama"

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def setup_weights(self, engines=None, model_weights=None, par_names=None):
        """
        Multi-model/param specific setup

        Parameters
        ----------
        engines : dict, optional
            Dictionary of engines to consider for the indices.
            If set to None, the engine provided to the object is assumed
            as the only model.
            If the engine of this object is not included, it is added.
            The default is None.
        model_weights : dict, optional
            Weights for every model in the model set 'engines'.
            The weights should be positive and sum to 1.
            If they are given as None, equal weights are assumed.
            The default is None.
        par_names : list, optional
            Parameters to run the analysis for.
            If set to None, all parameters are selected.
            The default is None.

        Returns
        -------
        Corrected engines, model_weights and parameters.

        """
        # Check the engines
        if engines is None:
            print("No engines is given, assuming single available model.")
            engines = {self.name: self.engine}
        else:
            if not isinstance(engines, dict):
                raise AttributeError("The given engines should be a dictionary!")
            if not self.name in engines:
                print("The current engine was not found in engines, adding now.")
                engines[self.name] = self.engine

        # Check the model weights
        if model_weights is None:
            print("No weights given, assuming equal weights.")
            n_engines = len(list(engines.keys()))
            model_weights = {e_name: 1 / n_engines for e_name in engines}
        if not isinstance(model_weights, dict):
            raise AttributeError("The given model weights should be a dictionary!")
        if list(model_weights.keys()) != list(engines.keys()):
            raise AttributeError(
                "The given engines and model_weights contain different keys!"
            )
        weights = np.array(list(model_weights.values()))
        if np.any(weights < 0):
            raise AttributeError("The model weights should be positive!")
        if np.sum(weights) != 1:
            raise AttributeError("The model weights should add to 1!")

        # Check the parameters - more checks done in par_to_par_id
        if par_names is None:
            print(
                "No parameters specified, running on all parameters in the set engine."
            )
            par_names = [[par] for par in self.par_names]

        return engines, model_weights, par_names

    # -------------------------------------------------------------------------
    def calc_indices(
        self,
        par_names: list = None,
        plot_type: str = "line",
        save: bool = True,
        plot: bool = True,
    ):
        """
        Calculate the AMA indices for a single model.
        This function returns the indices for the statistical moments
            mean, variance, skewness, curtosis
        for each of the given parameters or parameter groups
        as AMAE, AMAV, AMAs, AMAk.

        Indices published in:
            Dell'Oca, Aronne, Monica Riva, and Alberto Guadagnini.
            "Moment-based metrics for global sensitivity analysis of
            hydrological systems."
            Hydrology and Earth System Sciences 21.12 (2017): 6219-6234.

        Parameters
        ----------
        par_names : list, optional
            Names of the parameters to run the analysis for.
            The parameters have to belong to self.engine.ö
            It set to None, all parameters from the given engine are used individually.
            The default is None.

        Returns
        -------
        indices : dict
            Collection of the calculated indices.


        """
        # Currently only generates and stores first-order indices
        warnings.warn(
            "The calculation of AMA' indices currently only"
            " supports dependence on individual parameters."
        )

        engines, model_weights, par_names = self.setup_weights(None, None, par_names)

        # Generate mc samples and calculate the different moments
        mc_samples, mc_out = self.generate_mcsamples(engines, n_mc=self.n_mc)
        model_moment = self.calc_model_moment(mc_out)

        # Calculate indices for every parameter in the engine
        indices = {
            name: {
                str(par): {
                    key: np.zeros((len(self.x_values))) for key in self.engine.out_names
                }
                for par in par_names
            }
            for name in self.index_names
        }

        for _, parameter in tqdm(enumerate(par_names), "AMA indices, parameter"):
            # Conditional statistics for all except the parameter
            par_ids = self.par_to_parid(parameter)
            par_moment = self.calc_par_moment(
                mc_samples[self.name], mc_out[self.name], par_ids, self.n_bins, False
            )

            # For now, only run for individual parameters
            if len(par_ids) > 1:
                continue

            for m_id, moment in enumerate(list(self.moment_functions.keys())):
                for key in self.engine.out_names:
                    if key == "x_values":
                        continue
                    for t in range(len(self.x_values)):
                        model_stat = model_moment[moment][self.name][key][t]
                        par_cont = np.mean(
                            np.abs(model_stat - par_moment[moment][key][:, t])
                        )
                        denom = self.ama_denom(model_stat, None, par_cont, moment)

                        frac = model_weights[self.name] / denom
                        total = frac * par_cont

                        indices[self.index_names[m_id]][str(parameter)][key][t] = total
        if save:
            self.store_to_json(indices)
        if plot:
            self.plot(indices, plot_type=plot_type)
        return indices

    def calc_multimod_indices(
        self,
        engines: dict = None,
        model_weights: dict = None,
        par_names: list = None,
        ret_contrib: bool = False,
        plot_type: str = "line",
        save: bool = True,
        plot: bool = True,
    ):
        """
        Calculate the AMA indices for parameters in self.engine,
        in the context of the provided model set (engines).
        This function returns the indices for the statistical moments
            mean, variance, skewness, curtosis
        for each of the given parameters or parameter groups
        as AMAE, AMAV, AMAs, AMAk.

        Indices published in:
            Dell'Oca, A., M. Riva, and A. Guadagnini.
            "Global sensitivity analysis for multiple interpretive models
            with uncertain parameters."
            Water Resources Research 56.2 (2020): e2019WR025754.

        Parameters
        ----------
        engines : dict, optional
            Dictionary of all additional engines for a multimodel context.
            They keys are used as model names in the output dictionary.
            If set to None, only one available model is assumed.
            The default is None.
        model_weights : dict, optional
            Weights for each available model.
            If set to None, then equal weights are assumed.
            The default is None.
        par_names : list, optional
            Names of the parameters to run the analysis for.
            The parameters have to belong to self.engine.
            It set to None, all parameters from the given engine are used individually.
            The default is None.
        ret_contrib : bool, optional
            Toggles the return of the model and parameter contributions to the indices.
            If set to False, the full indices are returned.
            The default is False.

        Returns
        -------
        if ret_contrib = False:
            indices : dict
                The calculated overall AMA indices for every model.
        if ret_contrib == True:
            mod_cont, par_cont : dict, dict
                The model-choice and parameter-choice contributions.

        """
        # Currently only generates and stores first-order indices
        warnings.warn(
            "The calculation of AMA' indices currently only"
            " supports dependence on individual parameters."
        )

        engines, model_weights, par_names = self.setup_weights(
            engines, model_weights, par_names
        )

        # Generate mc samples and calculate the different moments
        mc_samples, mc_out = self.generate_mcsamples(engines, n_mc=self.n_mc)
        model_moment = self.calc_model_moment(mc_out)
        full_moment = self.calc_full_moment(model_moment, model_weights)

        # Calculate indices for parameters in self.engine
        model_contrib, parameter_contrib = {}, {}
        indices = {
            name: {
                str(par): {
                    key: np.zeros((len(self.x_values))) for key in self.engine.out_names
                }
                for par in par_names
            }
            for name in self.index_names
        }
        indices_mod = {
            name: {
                mod: {
                    key: np.zeros((len(self.x_values))) for key in self.engine.out_names
                }
                for mod in engines
            }
            for name in self.index_names
        }
        indices_par = {
            name: {
                mod: {
                    str(par): {
                        key: np.zeros((len(self.x_values)))
                        for key in self.engine.out_names
                    }
                    for par in par_names
                }
                for mod in engines
            }
            for name in self.index_names
        }

        for _, parameter in tqdm(
            enumerate(par_names), "Multimodel AMA indices, parameter"
        ):
            par_ids = self.par_to_parid(parameter)
            par_str = parameter[0] if len(parameter) == 1 else str(parameter)

            # For now, only run for individual parameters
            if len(par_ids) > 1:
                continue

            model_contrib[par_str] = {}
            parameter_contrib[par_str] = {}

            # Conditional statistics for all except the parameter
            samples, out = mc_samples[self.name], mc_out[self.name]
            par_moment = self.calc_par_moment(samples, out, par_ids, self.n_bins, False)

            # Calculate the indices
            for m_id, moment in enumerate(list(self.moment_functions.keys())):
                for key in out:
                    if key == "x_values":
                        continue
                    for t in range(out[key].shape[1]):
                        full_stat = full_moment[moment][key][t]
                        model_stat = model_moment[moment][self.name][key][t]

                        mod_cont = np.abs(full_stat - model_stat)
                        par_cont = np.mean(
                            np.abs(model_stat - par_moment[moment][key][:, t])
                        )
                        denom = self.ama_denom(full_stat, mod_cont, par_cont, moment)

                        frac = model_weights[self.name] / denom
                        total = frac * (mod_cont + par_cont)

                        indices[self.index_names[m_id]][str(parameter)][key][t] = total
                        indices_mod[self.index_names[m_id]][self.name][key][
                            t
                        ] = mod_cont
                        indices_par[self.index_names[m_id]][self.name][str(parameter)][
                            key
                        ][t] = par_cont

        if save:
            self.store_to_json(indices)
        if plot:
            self.plot(indices, plot_type=plot_type)

        # Return all the calculated values, if wanted
        if ret_contrib:
            return indices_mod, indices_par
        # Return only the final indices
        return indices

    def ama_denom(self, full_stat, mod_cont, par_cont, moment):
        """
        Build the AMA denominator

        Parameters
        ----------
        full_stat : float/double
            Value of the overall statistical moment.
        mod_cont : float/double
            Value of the model-specific index contribution.
        par_cont : float/double
            Value of the parameter-specific index contribution.
        moment : str
            Name of the moment that the other parameters belong to.

        """
        moment_id = list(self.moment_functions.keys()).index(moment)
        denom = np.abs(full_stat)

        if mod_cont == None:
            scale_diff = par_cont / denom
        elif par_cont == None:
            scale_diff = mod_cont / denom
        else:
            scale_diff = min(mod_cont / denom, par_cont / denom)
        if (
            moment_id % 2 == 0 and scale_diff > self.scale_diff
        ):  # and np.abs(full_stat) < 0.001:
            print(f"Found small value for {moment}: {denom}")
            denom = 1

        return denom

    def plot_par_moment(
        self,
        par_names=None,
        plt_name="",
        t_id=0,
        plt_type="line",
    ):
        """
        Calculate and visualize the output moments, conditioned on the model parameters.

        Parameters
        ----------
        par_names : list, optional
            List of the parameter sets to be plotted (list of lists).
            If set to None, then each parameter in the engine is considered separately.
            The default is None.
        out_key : string, optional
            The output key to visualize.
            The default is 'Z'
        t_id : int, optional
            The time index to visualize for.
            The default is 0.
        plt_type : str, optional
            Type of plot to generate, supports 'line' and 'hist'.
            The default is 'line'.

        """
        _, _, par_names = self.setup_weights(None, None, par_names)
        for key in self.engine.out_names:
            par_moments = []
            par_bins = []
            for _, par in tqdm(enumerate(par_names), f"Conditional moments for {key}"):
                par_ids = self.par_to_parid(par)
                mc_samples, mc_out = self.generate_mcsamples(
                    {"mod": self.engine}, n_mc=self.n_mc
                )
                mom, bins = self.calc_par_moment(
                    mc_samples["mod"],
                    mc_out["mod"],
                    par_ids,
                    self.n_bins,
                    return_bins=True,
                )
                par_moments.append(mom)
                par_bins.append(bins)

            for moment in self.moment_functions:
                fig, ax = plt.subplots(
                    nrows=len(par_names), ncols=1, figsize=(15, len(par_names) * 4)
                )
                fig.tight_layout()
                for p_id, par in enumerate(par_names):
                    if len(par_names) == 1:
                        axis = ax
                    else:
                        axis = ax[p_id]
                    if plt_type == "line":
                        axis.plot(
                            par_bins[p_id],
                            par_moments[p_id][moment][key][:, t_id],
                            linestyle="dashed",
                        )
                    elif plt_type == "hist":
                        axis.hist(
                            par_moments[p_id][moment][key][:, t_id],
                            linestyle="dashed",
                            bins=int(self.n_bins / 5),
                        )

                    axis.set_xlabel(par)
                    axis.set_ylabel(moment)
                fig.suptitle(key)
                if plt_name != "":
                    fig.savefig(
                        f"{self.out_dir}/conditional_{plt_name}_{moment}_{key}.{self.out_format}",
                        bbox_inches="tight",
                    )
                else:
                    fig.savefig(
                        f"{self.out_dir}/conditional_{moment}_{key}.{self.out_format}",
                        bbox_inches="tight",
                    )
                plt.close()
