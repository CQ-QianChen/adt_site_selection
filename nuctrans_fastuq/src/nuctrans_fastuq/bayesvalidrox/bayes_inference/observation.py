#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discrepancy class for measurement uncertainty.
"""

import os
import warnings
import numpy as np
import pandas as pd


class Observation:
    """
    Observation class to be used for Bayesian analysis.
    This class contains a set of measurements, and the related uncertain effects,
    given by an additive zero-mean Gaussian that describes the measurement error
    and model inaccuracy.

    Attributes
    ----------
    data : dict, optional
        Data from the observation.
        If not given, the data_file should be given.
        The default is None.
    data_file : str, optional
        Path to csv-file that contains the observation data in DataFrame format.
        If not given, the data should be set as a dictionary.
        The default is None.
    sigma2 : dict, optional
        Known residual variance \\(\\sigma^2\\), i.e. diagonal entries of the
        covariance matrix of the multivariate normal likelihood in case of
        given measurement uncertainty.
        If not given, it is initialized with all zeros.
        The default is None.
    out_names : list, optional
        List of output keys that the observation object should expect.
        If not given, the keys of the given data are used.
        The default is None.

    """

    def __init__(self, data=None, data_file=None, sigma2=None, out_names=None):
        # Inits
        self.data = data
        self.data_file = data_file
        self.sigma2 = sigma2

        # Other parameters
        self.out_names = out_names
        self.n_values = 0

    def build(self):
        """
        Check and build any needed parts of the object.
        """
        # Read in data if file given
        if self.data_file is not None:
            self.read_from_file()

        if isinstance(self.data, pd.DataFrame):  # and bool(self.data):
            self.data = self.data.to_dict(orient="list")
            self.data.pop("Unnamed: 0", None)

        # Check if data is given
        elif self.data is None or not isinstance(self.data, dict):
            raise AttributeError(
                "Please provide the observation data as a dictionary "
                "or pass the csv-file path to the attribute 'data_file'."
            )
        # Check for x_values for plotting
        if not "x_values" in self.data:
            raise AttributeError("No x_values in the observation data!")
        # Convert to numpy
        for key in self.data:
            self.data[key] = np.array(self.data[key])

        # Init sigma2 if not set
        if self.sigma2 is None:
            warnings.warn("No sigma2 given, initialize with all 0.")
            self.sigma2 = {}
            for key, meas in self.data.items():
                if key == "x_values":
                    continue
                self.sigma2[key] = np.zeros((meas.shape[0]), dtype=float)
        else:
            if isinstance(self.sigma2, pd.DataFrame):
                self.sigma2 = self.sigma2.to_dict(orient="list")
            # Note: no x_values in sigma2
            self.sigma2.pop("x_values", 0)
            for key, meas in self.sigma2.items():
                self.sigma2[key] = np.array(meas, dtype=float)

        # Check agreement of the given data with the output names
        meas_cols = list(self.data.keys())
        meas_out_cols = [key for key in self.data if key != "x_values"]
        if not self.out_names:
            self.out_names = meas_out_cols
        else:
            missing = [key for key in self.out_names if key not in meas_out_cols]
            if missing:
                raise ValueError(
                    f"Measured data is missing expected columns {missing}. "
                    f"Available columns: {meas_cols}"
                )

        # Check agreement of the given sigma2 with the output names
        sig_cols = list(self.sigma2.keys())
        sig_out_cols = [key for key in self.sigma2 if key != "x_values"]
        missing = [key for key in self.out_names if key not in sig_out_cols]
        if missing:
            raise ValueError(
                f"Measured sigma2 is missing expected columns {missing}. "
                f"Available columns: {sig_cols}"
            )

        # Extract the total number of measurement points per output (non-NaN)
        # So far this only supports single sets of observations!
        per_out_counts = {c: self.data[c].shape[0] for c in self.out_names}
        self.n_values = int(np.sum(list(per_out_counts.values())))

    def read_from_file(self):
        """
        Read the observations from a given file.
        """
        file_path = os.path.join(os.getcwd(), self.data_file)
        data = pd.read_csv(file_path, delimiter=",").to_dict(orient="list")
        data.pop("Unnamed: 0", None)
        self.data = data
