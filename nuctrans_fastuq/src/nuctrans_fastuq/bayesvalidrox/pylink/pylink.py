#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calls to the model and evaluations
"""

from dataclasses import dataclass

import os
import shutil
import zipfile
import time
import multiprocessing
from functools import partial
import h5py
import numpy as np
import pandas as pd
import tqdm
import umbridge

# from multiprocessing import get_context
from multiprocess import get_context


def within_range(out, minout, maxout):
    """
    Checks if all the values in out lie between minout and maxout

    Parameters
    ----------
    out : array or list
        Data to check against range
    minout : int
        Lower bound of the range
    maxout : int
        Upper bound of the range

    Returns
    -------
    inside : bool
        True if all values in out are in the specified range

    """
    try:
        out = np.array(out)
    except Exception as exc:
        raise AttributeError(
            "The given values should be a 1D array, but are not"
        ) from exc
    if out.ndim != 1:
        raise AttributeError("The given values should be a 1D array, but are not")

    if minout > maxout:
        raise ValueError(
            "The lower and upper bounds do not form a valid range, they might be switched"
        )

    inside = False
    if (out > minout).all() and (out < maxout).all():
        inside = True
    return inside


class PyLinkForwardModel:
    """
    A forward model binder

    This calss serves as a code wrapper. This wrapper allows the execution of
    a third-party software/solver within the scope of BayesValidRox.

    Attributes
    ----------
    link_type : str
        The type of the wrapper. The default is `'pylink'`. This runs the
        third-party software or an executable using a shell command with given
        input files.
        Second option is `'function'` which assumed that model can be run using
        a function written separately in a Python script.
    name : str
        Name of the model.
    py_file : str
        Python file name without `.py` extension to be run for the `'function'`
        wrapper. Note that the name of the python file and that of the function
        must be simillar. This function must recieve the parameters in an array
        of shape `(n_samples, n_params)` and returns a dictionary with the
        x_values and output arrays for given output names.
    func_args : dict
        Additional arguments for the python file. The default is `{}`.
    shell_command : str
        Shell command to be executed for the `'pylink'` wrapper.
    input_file : str or list
        The input file to be passed to the `'pylink'` wrapper.
    input_template : str or list
        A template input file to be passed to the `'pylink'` wrapper. This file
        must be a copy of `input_file` with `<Xi>` place holder for the input
        parameters defined using `inputs` class, with i being the number of
        parameter. The file name ending should include `.tpl` before the actual
        extension of the input file, for example, `params.tpl.input`.
    aux_file : str or list
        The list of auxiliary files needed for the `'pylink'` wrapper.
    exe_path : str
        Execution path if you wish to run the model for the `'pylink'` wrapper
        in another directory. The default is `None`, which corresponds to the
        currecnt working directory.
    output_file_names : list of str
        List of the name of the model output text files for the `'pylink'`
        wrapper.
    output_names : list of str
        List of the model outputs to be used for the analysis.
    output_parser : str
        Name of the model parser file (without `.py` extension) that recieves
        the `output_file_names` and returns a 2d-array with the first row being
        the x_values, e.g. x coordinates or time and the rest of raws pass the
        simulation output for each model output defined in `output_names`. Note
        that again here the name of the file and that of the function must be
        the same.
    multi_process: bool
        Whether the model runs to be executed in parallel for the `'pylink'`
        wrapper. The default is `True`.
    n_cpus: int
        The number of cpus to be used for the parallel model execution for the
        `'pylink'` wrapper. The default is `None`, which corresponds to all
        available cpus.
    meas_file : str
        The name of the measurement text-based file. This file must contain
        x_values as the first column and one column for each model output. The
        default is `None`. Only needed for the Bayesian Inference.
    meas_file_valid : str
        The name of the measurement text-based file for the validation. The
        default is `None`. Only needed for the validation with Bayesian
        Inference.
    mc_ref_file : list
        The names of the csv files for the Monte-Carlo reference values, one
        file each for mean and standard deviation values.
        Each file must contain one column per output key.
    obs_dict : dict
        A dictionary containing the measurement text-based file. It must
        contain `x_values` as the first item and one item for each model output
        . The default is `{}`. Only needed for the Bayesian Inference.
    obs_dict_valid : dict
        A dictionary containing the validation measurement text-based file. It
        must contain `x_values` as the first item and one item for each model
        output. The default is `{}`.
    mc_ref_dict : dict
        A dictionary containing the Monte-Carlo reference (mean and standard
        deviation) values. It must contain `x_values` as the first item and
        `mean` as the second item and `std` as the third. The default is `{}`.
        This is only available for one output.
    """

    # Nested class
    @dataclass
    class OutputData:
        """
        Output description.
        """

        parser: str = ""
        names: list = None
        file_names: list = None

    def __init__(
        self,
        link_type="pylink",
        name=None,
        py_file=None,
        func_args=None,
        shell_command="",
        input_file=None,
        input_template=None,
        aux_file=None,
        exe_path="",
        output_file_names=None,
        output_names=None,
        output_parser="",
        multi_process=True,
        n_cpus=None,
        meas_file=None,
        meas_file_valid=None,
        mc_ref_file=None,
        obs_dict=None,
        obs_dict_valid=None,
        mc_ref_dict=None,
        out_dir="",
    ):
        self.link_type = link_type
        self.name = name
        self.shell_command = shell_command
        self.py_file = py_file
        self.func_args = {} if func_args is None else func_args
        self.input_file = input_file
        self.input_template = input_template
        self.aux_file = aux_file
        self.exe_path = exe_path
        self.multi_process = multi_process
        self.n_cpus = n_cpus
        self.output = self.OutputData(
            parser=output_parser,
            names=[] if output_names is None else output_names,
            file_names=[] if output_file_names is None else output_file_names,
        )
        self.n_outputs = len(self.output.names)
        self.meas_file = meas_file
        self.meas_file_valid = meas_file_valid
        self.mc_ref_file = mc_ref_file
        self.observations = {} if obs_dict is None else obs_dict
        self.observations_valid = {} if obs_dict_valid is None else obs_dict_valid
        self.mc_reference = {} if mc_ref_dict is None else mc_ref_dict
        self.out_dir = out_dir

        self.n_obs = None
        self.n_obs_valid = None

        # um_bridge settings
        self.host = None
        self.model_link = None
        self.modelparams = None
        self.x_values = None

    # -------------------------------------------------------------------------
    def read_mcref(self):
        """
        # TODO: rename into 'read_mcref'?
        Reads/prepares the MC reference for estimating surrogate errors.
        The mc-reference should be a dict with keys 'mean' and 'std',
        with sub-dictionaries that contain the reference values for each output key.

        Returns
        -------
        DataFrame
            A dataframe with the calibration data.

        """
        # If form is incorrect
        if len(list(self.mc_reference)) == 0:
            # Read in if file given
            if (self.mc_ref_file is not None) and (
                list(self.mc_ref_file.keys()) == ["mean", "std"]
            ):
                file_path = os.path.join(os.getcwd(), self.mc_ref_file["mean"])
                data = pd.read_csv(file_path, delimiter=",").to_dict(orient="list")
                data.pop("Unnamed: 0", None)
                self.mc_reference["mean"] = data

                file_path = os.path.join(os.getcwd(), self.mc_ref_file["std"])
                data = pd.read_csv(file_path, delimiter=",").to_dict(orient="list")
                data.pop("Unnamed: 0", None)
                self.mc_reference["std"] = data
            else:
                raise AttributeError(
                    "Please provide the mc reference as a dictionary via the attribute "
                    "'mc_reference' or pass the csv-file paths to the attribute 'mc_ref_file'."
                )
        # Check what is in the dict
        if list(self.mc_reference.keys()) != ["mean", "std"]:
            raise AttributeError(
                "The keys in the mc-reference should be ['mean', 'std']."
            )
        # If all is correct, return
        return self.mc_reference

    def read_obs_file(self, file_path):
        """Helper-function to read the obervations"""
        data = pd.read_csv(file_path, delimiter=",").to_dict(orient="list")
        data.pop("Unnamed: 0", None)
        return data

    # -------------------------------------------------------------------------
    def read_output(self):
        """
        Reads the the parser output file and returns it as an
         executable function. It is required when the models returns the
         simulation outputs in csv files.

        Returns
        -------
        Output : func
            Output parser function.

        """
        output_func_name = self.output.parser

        output_func = getattr(__import__(output_func_name), output_func_name)

        file_names = []
        for file in self.output.file_names:
            file_names.append(os.path.join(self.exe_path, file))
        try:
            output = output_func(self.name, file_names)
        except TypeError:
            output = output_func(file_names)
        return output

    # -------------------------------------------------------------------------
    def update_input_params(self, new_input_file, param_set):
        """
        Finds this pattern with <X1> in the new_input_file and replace it with
         the new value from the array param_sets.

        Parameters
        ----------
        new_input_file : list
            List of the input files with the adapted names.
        param_set : array of shape (n_params)
            Parameter set.

        Returns
        -------
        None.

        """
        n_of_pa = param_set.shape[0]
        text_to_search_list = [f"<X{i+1}>" for i in range(n_of_pa)]

        for filename in new_input_file:
            # Read in the file
            with open(filename, "r") as file:
                filedata = file.read()

            # Replace the target string
            for text_to_search, params in zip(text_to_search_list, param_set):
                filedata = filedata.replace(text_to_search, f"{params:0.4e}")

            # Write the file out again
            with open(filename, "w") as file:
                file.write(filedata)

    # -------------------------------------------------------------------------
    def run_command(self, command, output_file_names):
        """
        Runs the execution command given by the user to run the given model.
        It checks if the output files have been generated. If yes, the jobe is
        done and it extracts and returns the requested output(s). Otherwise,
        it executes the command again.

        Parameters
        ----------
        command : str
            The shell command to be executed.
        output_file_names : list
            Name of the output file names.

        Returns
        -------
        simulation_outputs : array of shape (n_obs, n_outputs)
            Simulation outputs.

        """

        # Check if simulation is finished
        while True:
            time.sleep(3)
            files = os.listdir(".")
            if all(elem in files for elem in output_file_names):
                break
            # Run command
            print(f"Running command: ./../{command}")

            process = os.system(f"./../{command}")
            if process != 0:

                print(f"Command failed with exit code: {process}")

                print("\nMessage 1:")
                print(
                    f"\tIf the value of '{process}' is a non-zero value"
                    ", then compilation problems occur \n" % process
                )
        os.chdir("..")

        # Read the output
        simulation_outputs = self.read_output()

        return simulation_outputs

    # -------------------------------------------------------------------------
    def run_forwardmodel(self, xx):
        """
        This function creates subdirectory for the current run and copies the
        necessary files to this directory and renames them. Next, it executes
        the given command.

        Parameters
        ----------
        xx : tuple
            A tuple including parameter set, simulation number and key string.

        Returns
        -------
        output : array of shape (n_outputs+1, n_obs)
            An array passed by the output paraser containing the x_values as
            the first row and the simulations results stored in the the rest of
            the array.

        """
        c_points, run_no, key_str = xx

        # Handle if only one imput file is provided
        if not isinstance(self.input_template, list):
            self.input_template = [self.input_template]
        if not isinstance(self.input_file, list):
            self.input_file = [self.input_file]

        new_input_file = []
        # Loop over the InputTemplates:
        for in_temp in self.input_template:
            if "/" in in_temp:
                in_temp = in_temp.split("/")[-1]
            new_input_file.append(
                in_temp.split(".tpl")[0]
                + key_str
                + f"_{run_no+1}"
                + in_temp.split(".tpl")[1]
            )

        # Create directories
        newpath = self.name + key_str + f"_{run_no+1}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # Copy the necessary files to the directories
        for in_temp in self.input_template:
            # Input file(s) of the model
            shutil.copy2(in_temp, newpath)
        # Auxiliary file
        if self.aux_file is not None:
            shutil.copy2(self.aux_file, newpath)  # Auxiliary file

        # Rename the Inputfile and/or auxiliary file
        os.chdir(newpath)
        for input_tem, input_file in zip(self.input_template, new_input_file):
            if "/" in input_tem:
                input_tem = input_tem.split("/")[-1]
            os.rename(input_tem, input_file)

        # Update the parametrs in Input file
        self.update_input_params(new_input_file, c_points)

        # Update the user defined command and the execution path
        try:
            new_command = self.shell_command.replace(
                self.input_file[0], new_input_file[0]
            )
            new_command = new_command.replace(self.input_file[1], new_input_file[1])
        except:
            new_command = self.shell_command.replace(
                self.input_file[0], new_input_file[0]
            )
        # Set the exe path if not provided
        if not bool(self.exe_path):
            self.exe_path = os.getcwd()

        # Run the model
        output = self.run_command(new_command, self.output.file_names)

        return output

    # -------------------------------------------------------------------------
    def run_model_parallel(
        self,
        c_points,
        prev_run_no=0,
        key_str="",
        mp=True,
        verbose=True,
        store_hdf5=True,
    ):
        """
        Runs model simulations. If mp is true (default), then the simulations
         are started in parallel.

        Parameters
        ----------
        c_points : array of shape (n_samples, n_params)
            Collocation points (training set).
        prev_run_no : int, optional
            Previous run number, in case the sequential design is selected.
            The default is `0`.
        key_str : str, optional
            A descriptive string for validation runs. The default is `''`.
        mp : bool, optional
            Multiprocessing. The default is `True`.
        verbose: bool, optional
            Verbosity. The default is `True`.

        Returns
        -------
        all_outputs : dict
            A dictionary with x values (time step or point id) and all outputs.
            Each key contains an array of the shape `(n_samples, n_obs)`.
        new_c_points : array
            Updated collocation points (training set). If a simulation does not
            executed successfully, the parameter set is removed.

        """

        # Initilization
        n_c_points = len(c_points)
        all_outputs = {}
        model_function = None

        # If the link type is UM-Bridge, then no parallel needs to be started from here
        if self.link_type.lower() == "umbridge":
            if not hasattr(self, "x_values"):
                raise AttributeError(
                    "For model type `umbridge` the attribute `x_values` "
                    "needs to be set for the model!"
                )
            # Init model
            # model = umbridge.HTTPModel('http://localhost:4242', 'forward')
            self.model_link = umbridge.HTTPModel(self.host, "forward")
            model_function = self.um_bridge_model

        # Extract the function
        if self.link_type.lower() == "function":
            # Prepare the function
            model_function = getattr(__import__(self.py_file), self.py_file)
        # ---------------------------------------------------------------
        # -------------- Multiprocessing with Pool Class ----------------
        # ---------------------------------------------------------------
        # Start a pool with the number of CPUs
        if self.n_cpus is None:
            n_cpus = multiprocessing.cpu_count()
        else:
            n_cpus = self.n_cpus

        # Run forward model
        if n_c_points == 1 or not mp:
            if n_c_points == 1:
                if (
                    self.link_type.lower() == "function"
                    or self.link_type.lower() == "umbridge"
                ):
                    group_results = model_function(c_points, **self.func_args)
                else:
                    group_results = self.run_forwardmodel(
                        (c_points[0], prev_run_no, key_str)
                    )
            else:
                for i in range(c_points.shape[0]):
                    if i == 0:
                        if (
                            self.link_type.lower() == "function"
                            or self.link_type.lower() == "umbridge"
                        ):
                            group_results = model_function(
                                np.array([c_points[0]]), **self.func_args
                            )
                        else:
                            group_results = self.run_forwardmodel(
                                (c_points[0], prev_run_no, key_str)
                            )
                        for key in group_results:
                            if key != "x_values":
                                group_results[key] = [group_results[key]]
                    else:
                        if (
                            self.link_type.lower() == "function"
                            or self.link_type.lower() == "umbridge"
                        ):
                            res = model_function(
                                np.array([c_points[i]]), **self.func_args
                            )
                        else:
                            res = self.run_forwardmodel(
                                (c_points[i], prev_run_no, key_str)
                            )
                        for key in res:
                            if key != "x_values":
                                group_results[key].append(res[key])

                for key in group_results:
                    if key != "x_values":
                        group_results[key] = np.array(group_results[key])

        elif self.multi_process or mp:
            with get_context("spawn").Pool(n_cpus) as p:
                # with multiprocessing.Pool(n_cpus) as p:

                if (
                    self.link_type.lower() == "function"
                    or self.link_type.lower() == "umbridge"
                ):
                    imap_var = p.imap(
                        partial(model_function, **self.func_args),
                        c_points[:, np.newaxis],
                    )
                else:
                    args = zip(
                        c_points,
                        [prev_run_no + i for i in range(n_c_points)],
                        [key_str] * n_c_points,
                    )
                    imap_var = p.imap(self.run_forwardmodel, args)

                if verbose:
                    desc = f"Running forward model {key_str}"
                    group_results = list(
                        tqdm.tqdm(imap_var, total=n_c_points, desc=desc)
                    )
                else:
                    group_results = list(imap_var)

        # Check for NaN
        for _, var in enumerate(self.output.names):
            outputs = None
            # If results are given as one dictionary
            if isinstance(group_results, dict):
                outputs = np.asarray(group_results[var])
            # If results are given as list of dictionaries
            elif isinstance(group_results, list):
                outputs = np.asarray(
                    [item[var] for item in group_results], dtype=np.float64
                )
            nan_idx = np.unique(np.argwhere(np.isnan(outputs))[:, 0])
            new_c_points = np.delete(c_points, nan_idx, axis=0)
            all_outputs[var] = np.atleast_2d(np.delete(outputs, nan_idx, axis=0))

        # Print the collocation points whose simulations crashed
        if len(nan_idx) != 0:
            print("\n")
            print("*" * 20)
            print(
                "\nThe following parameter sets have been removed:\n", c_points[nan_idx]
            )
            print("\n")
            print("*" * 20)

        # Save time steps or x-values
        if isinstance(group_results, dict):
            all_outputs["x_values"] = group_results["x_values"]
        elif any(isinstance(i, dict) for i in group_results):
            all_outputs["x_values"] = group_results[0]["x_values"]

        # Store simulations in a hdf5 file
        if store_hdf5:
            # print(c_points, all_outputs)
            # self.delete_hdf5(f"ExpDesign_{self.name}.hdf5")
            self._store_simulations(
                c_points, all_outputs, nan_idx, key_str, prev_run_no
            )

        return all_outputs, new_c_points

    def um_bridge_model(self, params):
        """
        Function that calls a um_bridge model and transforms its output into the
        shape expected for the surrogate.

        Parameters
        ----------
        params : 2d np.array, shape (#samples, #params)
            The parameter values for which the model is run.

        Returns
        -------
        dict
            The transformed model outputs.

        """
        # Run the model
        # out = np.array(model(np.ndarray.tolist(params), {'level':0}))
        out = np.array(self.model_link(np.ndarray.tolist(params), self.modelparams))

        # Sort into dict
        out_dict = {}
        cnt = 0
        for key in self.output.names:
            out_dict[key] = out[:, cnt]
            cnt += 1
        out_dict["x_values"] = self.x_values
        return out_dict

    # -------------------------------------------------------------------------
    def delete_hdf5(self, filename: str) -> None:
        """
        Deletes the hdf5 file if it exists.

        Parameters
        ----------
        filename : str
            Name of the file to be deleted.

        Returns
        -------
        None.

        """
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Deleted {filename}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}")

    # -------------------------------------------------------------------------
    def _store_simulations(self, c_points, all_outputs, nan_idx, key_str, prev_run_no):
        """
        Writes the results of model runs into an hdf5-file.

        Parameters
        ----------
        c_points : TYPE
            DESCRIPTION.
        all_outputs : TYPE
            DESCRIPTION.
        nan_idx : TYPE
            DESCRIPTION.
        key_str : TYPE
            DESCRIPTION.
        prev_run_no : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Create hdf5 metadata´
        if key_str == "":
            hdf5file = f"ExpDesign_{self.name}.hdf5"
        else:
            hdf5file = f"ValidSet_{self.name}.hdf5"
        # Deletes any existing hdf5 file
        self.delete_hdf5(hdf5file)

        with h5py.File(hdf5file, "a") as file:
            # Save time steps or x-values
            if isinstance(all_outputs["x_values"], dict):
                grp_x_values = file.create_group("x_values/")
                for _, var in enumerate(self.output.names):
                    grp_x_values.create_dataset(var, data=all_outputs["x_values"][var])
            else:
                file.create_dataset("x_values", data=all_outputs["x_values"])

            # ---------- Save outputs ----------
            for _, var in enumerate(self.output.names):
                grp_y = file.create_group("EDY/" + var)

                if prev_run_no == 0 and key_str == "":
                    grp_y.create_dataset(f"init_{key_str}", data=all_outputs[var])
                else:
                    try:
                        old_ed_y = np.array(file[f"EDY/{var}/adaptive_{key_str}"])
                        del file[f"EDY/{var}/adaptive_{key_str}"]
                        data = np.vstack((old_ed_y, all_outputs[var]))
                    except KeyError:
                        data = all_outputs[var]
                    grp_y.create_dataset("adaptive_" + key_str, data=data)

                if prev_run_no == 0 and key_str == "":
                    grp_y.create_dataset(f"New_init_{key_str}", data=all_outputs[var])
                else:
                    try:
                        name = f"EDY/{var}/New_adaptive_{key_str}"
                        old_ed_y = np.array(file[name])
                        del file[f"EDY/{var}/New_adaptive_{key_str}"]
                        data = np.vstack((old_ed_y, all_outputs[var]))
                    except KeyError:
                        data = all_outputs[var]
                    grp_y.create_dataset(f"New_adaptive_{key_str}", data=data)

            # ---------- Save CollocationPoints ----------
            new_c_points = np.delete(c_points, nan_idx, axis=0)
            grp_x = file.create_group("EDX")
            if prev_run_no == 0 and key_str == "":
                grp_x.create_dataset("init_" + key_str, data=c_points)
                if len(nan_idx) != 0:
                    grp_x.create_dataset("New_init_" + key_str, data=new_c_points)

            else:
                try:
                    name = f"EDX/adaptive_{key_str}"
                    old_colloc_points = np.array(file[name])
                    del file[f"EDX/adaptive_{key_str}"]
                    data = np.vstack((old_colloc_points, new_c_points))
                except KeyError:
                    data = new_c_points
                grp_x.create_dataset("adaptive_" + key_str, data=data)

                if len(nan_idx) != 0:
                    try:
                        name = f"EDX/New_adaptive_{key_str}"
                        old_colloc_points = np.array(file[name])
                        del file[f"EDX/New_adaptive_{key_str}"]
                        data = np.vstack((old_colloc_points, new_c_points))
                    except KeyError:
                        data = new_c_points
                    grp_x.create_dataset("New_adaptive_" + key_str, data=data)

        # Close h5py file
        file.close()

    # -------------------------------------------------------------------------
    def zip_subdirs(self, dir_name, key):
        """
        Zips all the files containing the key(word).

        Parameters
        ----------
        dir_name : str
            Directory name.
        key : str
            Keyword to search for.

        Returns
        -------
        None.

        """
        # setup file paths variable
        dir_list = []
        file_paths = []

        # Read all directory, subdirectories and file lists
        dir_path = os.getcwd()

        for root, directories, files in os.walk(dir_path):
            for directory in directories:
                # Create the full file_path by using os module.
                if key in directory:
                    folder_path = os.path.join(dir_path, directory)
                    dir_list.append(folder_path)

        # Loop over the identified directories to store the file paths
        for direct_name in dir_list:
            for root, directories, files in os.walk(direct_name):
                for filename in files:
                    # Create the full file_path by using os module.
                    file_path = os.path.join(root, filename)
                    file_paths.append("." + file_path.split(dir_path)[1])

        # writing files to a zipfile
        if len(file_paths) != 0:
            zip_file = zipfile.ZipFile(dir_name + ".zip", "w")
            with zip_file:
                # writing each file one by one
                for file in file_paths:
                    zip_file.write(file)

            file_paths = [path for path in os.listdir(".") if key in path]

            for path in file_paths:
                shutil.rmtree(path)

            print("\n")
            print(f"{dir_name}.zip has been created successfully!\n")
