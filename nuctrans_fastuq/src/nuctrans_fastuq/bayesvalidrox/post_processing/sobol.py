#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for calculation of Sobol' indices.
"""

import os
import copy
from itertools import combinations
from tqdm import tqdm
import numpy as np

from .gsa import GSA


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


class Sobol(GSA):
    """
    This class provides functions for calculating Sobol' indices on trained engines.

    Parameters
    ----------
    engine : obj
        Trained Engine object, is expected to contain a trained MetaModel object.
    name : string
        Name to be used for saving the generated files.
        The default is 'calib'. # TODO: set another default name instead?
    out_dir : string
        Output directory in which the Sobol results are placed.
        # TODO: this folder structure ok?
        The results are contained in a subfolder '/Outputs_GSA_name/sobol'
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
        self, engine, 
        name="calib", 
        out_dir="", 
        out_format="pdf", 
        n_mc=1000, 
        n_bins=100
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
        self.setup()

    def setup(self):
        """
        Run the setup - use GSA setup and build output directory.
        """
        super().setup()

        # Adapt output directory
        if self.out_dir[:2] != "./":
            self.out_dir = f"./{self.out_dir}"

        if self.out_dir[-6:] != "/sobol":
            self.out_dir = f"{self.out_dir}/sobol"

        # Create output folder for the plots
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    # -------------------------------------------------------------------------
    def calc_indices(
        self,
        plot_type: str = "line",
        save: bool = True,
        plot: bool = True,
    ):
        """
        Visualizes and writes out Sobol' and Total Sobol' indices of the trained engine.
        One file is created for each index and output key.

        If the engine has a PCE-type metamodel, the indices are calculated directly
        from the coefficients.
        Otherwise MC-sampling is used to estimate the moments.

        Parameters
        ----------
        plot_type : str, optional
            Plot type, supports 'line' for lineplots and 'bar' for barplots.
            The default is `line`.
            Bar chart can be selected by `bar`.
        save : bool, optional
            Write out the inidces as csv files if set to True.
            The default is True.
        plot : bool, optional
            Visualizes the calculated indices if set to True.
            The default is True.

        Raises
        ------
        AttributeError
            MetaModel in given Engine needs to be of type 'pce' or 'apce'.
        AttributeError
            Plot-type must be 'line' or 'bar'.

        Returns
        -------
        sobol_all : dict
            All possible Sobol' indices for the given metamodel.
        total_sobol_all : dict
            All Total Sobol' indices for the given metamodel.

        """
        self.setup()

        # TODO: move to plotting function?
        if plot_type not in ["line", "bar"]:
            raise AttributeError("The wanted plot type is not supported.")

        # For PCE/aPCE calculate directly from the coefficients
        metamod = self.engine.meta_model
        if hasattr(metamod, "sobol"):
            # if metamod.sobol is None:
            indices = metamod.calculate_sobol(y_train=self.engine.exp_design.y)
        else:
            # Calculate from MC-samples
            indices = self.calc_indices_general()

        # Save and visualize indices
        if save:
            self.store_to_json(indices)
        if plot:
            self.plot(indices, plot_type)
        return indices

    def calc_indices_general(self):
        """
        Calculation of Sobol' indices for the given engine.
        This calculation is independent of the engine/surrogate type.
        """
        # Generate mc samples
        mc_samples, mc_out = self.generate_mcsamples(
            {"model": self.engine}, n_mc=self.n_mc
        )
        full_var = self.calc_model_moment(mc_out)["var"]["model"]

        # Generate the set of possible parameter combination
        param_list = calc_param_combinations(self.par_names)

        # Init the storage dict
        max_order = 1 #len(self.par_names)
        i_names = [f"Sobol_{i+1}" for i in range(max_order)]
        out_dict = {
            key: np.zeros((1, len(self.x_values))) for key in self.engine.out_names
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
        for i in tqdm(range(max_order), "Sobol indices, Order"):
            order = i + 1
            s_key = f"Sobol_{order}"
            for pars in param_list[order]:
                par_id = self.par_to_parid(list(pars))
                cond_metrics = self.calc_par_moment(
                    mc_samples["model"], mc_out["model"], par_id, n_bins=self.n_bins
                )
                cond_metrics = cond_metrics["mean"]

                for out_key in self.engine.out_names:
                    for t in range(len(self.x_values)):
                        # Sobol indices
                        s_idx = (
                            self.moment_functions["var"](cond_metrics[out_key][:, t])
                            / full_var[out_key][t]
                        )
                        # TODO: first, third order seems correct, second is off
                        indices[s_key][str(pars)][out_key][:, t] = s_idx

                        # Add to TotalSobol
                        # TODO: look correct, only errors from wrong sobol
                        for par_id2, par in enumerate(self.par_names):
                            if par_id2 in par_id:
                                indices["TotalSobol"][str([par])][out_key][
                                    :, t
                                ] += s_idx

        return indices
