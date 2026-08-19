# Import modules 
import scipy
import scipy.stats
from scipy.stats import norm, truncnorm, lognorm, beta, uniform
import scipy.stats.qmc as qmc
import yaml
import numpy as np
from pathlib import Path
import os
import sys
import argparse
import gzip
import pickle
import h5py


def geometry_dist(**kwargs):
    """
    Function to sample distribution the boundary layer of each rock unit.

    Args:
        **kwargs: additional keyword arguments for geometry uncertainties. 
            units (dict): dictionary containing names and uncertain ranges for boundary layers.

    Returns:
        list(dict): a list of dictionaries containing names and sample distribution for boundary layers.
    """
    geometry_uncertainties = []
    # Check if 'units' geometry configurations are provided
    if kwargs['uncertain_parameters'].get('geometry') is not None:
        for rock_interface, bounds in kwargs['uncertain_parameters']['geometry'].items():
            rock_interface_file = os.path.join(kwargs.get("path_to_geometry_data"), rock_interface+".yaml")
            with open(rock_interface_file, "r", encoding="utf-8") as f:
                rock_interface_depths = yaml.safe_load(f)
            for bound in bounds:
                # Retrieve the probability distribution string from the YAML file
                sampled_data_rvs = rock_interface_depths[bound][0]["probability_distribution"]["sampled_data"]
                # Remove the .rvs(...) call
                geometry_uncertainties.append({
                    'category': 'geometry',
                    'layer': bound,
                    'prop': 'rock_interface',
                    'dist': eval(sampled_data_rvs[:sampled_data_rvs.index(".rvs(")])
                })
        return geometry_uncertainties
    else:
        raise ValueError("No uncertainties of medium properties specified.")

def medium_properties_dist(**kwargs):
    """
    Function to generate sample distribution for medium properties.

    Args:
        **kwargs: additional keyword arguments for medium properties uncertainties. 
            rock_data (dict): dictionary containing rock units and their corresponding uncertain properties.

    Returns:
        list(dict): a list of dictionaries containing sample distribution medium properties.
    """
    medium_props_samples = []
    if kwargs["uncertain_parameters"].get("rock_data") is not None:
        for rock_unit, uncertain_rock_properties in kwargs["uncertain_parameters"]["rock_data"].items():
            rock_unit_props_file = os.path.join(kwargs.get("path_to_rock_data"), rock_unit+".yaml")
            with open(rock_unit_props_file, "r", encoding="utf-8") as f:
                rock_unit_props = yaml.safe_load(f)

            for rp in uncertain_rock_properties:
                try:
                    # Retrieve the probability distribution string from the YAML file
                    sampled_data_rvs = rock_unit_props[rp][0]["probability_distribution"]["sampled_data"]
                    # Remove the .rvs(...) call
                    if sampled_data_rvs:
                        medium_props_samples.append({
                            'category': 'rock_data',
                            'layer': rock_unit,
                            'prop': rp,
                            'dist': eval(sampled_data_rvs[:sampled_data_rvs.index(".rvs(")])
                        })
                    else:  # sampled_data_rvs is None
                        raise ValueError(
                            f"No probability distribution was provided for property '{rp}' "
                            f"in rock unit '{rock_unit}'. "
                            f"This property is treated as constant with its specified deterministic value {rock_unit_props[rp][0]['value']}. "
                            f"If you want to consider uncertainties for this property, you must provide "
                            f"a probability distribution explicitly in file {rock_unit_props_file}."
                        )
                except KeyError:
                    raise KeyError(f"Property {rp} not found for rock unit {rock_unit}. The available properties are: {list(rock_unit_props.keys())}")
        return medium_props_samples
    else:
        raise ValueError("No uncertainties of medium properties specified.")

def nuclide_properties_dist(nuclide_prop, **kwargs):
    """
    Function to generate sample distribution for nuclide properties.

    Args:
        nuclide_prop (str): the nuclide property to sample (e.g., 'nuclide_sorption_data', 'nuclide_water_diffusivity_data').
        **kwargs: additional keyword arguments for nuclide properties uncertainties. 
            nuclide_sorption_data (dict): dictionary containing rock units, nuclides, and their corresponding uncertain properties.
    Returns:
        list(dict): a list of dictionaries containing sample distribution nuclide properties.
    """
    nuclides_props_samples = []
    if nuclide_prop == "nuclide_sorption_data":
        path_to_nuclide_data = "path_to_sorption_data"
    if nuclide_prop == "nuclide_water_diffusivity_data":
        path_to_nuclide_data = "path_to_diffusivity_data"

    if kwargs["uncertain_parameters"].get(nuclide_prop) is not None:
        for rock_unit, nuclides_list in kwargs["uncertain_parameters"][nuclide_prop].items():
            
            for nuclide in nuclides_list:
                
                nuclide_file = os.path.join(kwargs.get(path_to_nuclide_data), rock_unit+".yaml")
                with open(nuclide_file, "r", encoding="utf-8") as f:
                    nuclide_data = yaml.safe_load(f)

                # Get samples for each property from the PDF string
                try:
                    # Locate the specific nuclide item in the list for the given property
                    np_rvs = nuclide_data[nuclide][0]["probability_distribution"]["sampled_data"]
                    if np_rvs:
                        nuclides_props_samples.append({
                            'category': 'nuclide_properties',
                            'layer': rock_unit,
                            'prop': (nuclide, nuclide_prop),
                            'dist': eval(np_rvs[:np_rvs.index(".rvs(")])
                        })
                    else:  # np_rvs is None
                        raise ValueError(
                            f"No probability distribution was provided for property '{nuclide_prop}' "
                            f"of nuclide '{nuclide}' in rock unit '{rock_unit}'. "
                            f"This property is treated as constant with its specified deterministic value {nuclide_data[nuclide][0]['value']}. "
                            f"If you want to consider uncertainties for this property, you must provide "
                            f"a probability distribution explicitly in file {nuclide_file}."
                        )

                except KeyError:
                    raise KeyError(f"Property {nuclide_prop} for nuclide {nuclide} not found for rock unit {rock_unit}. The available properties are: {list(nuclide_data.keys())}")
                    
        return nuclides_props_samples
    else:
        raise ValueError("No uncertainties of nuclide properties specified.")

# Unified Sample Parameters Coordinator (SciPy & SciPy QMC)

def sample_parameters(uncertain_parameters):
    """
    Unified function to sample parameters either randomly using SciPy or 
    quasi-randomly using SciPy QMC (Sobol or Latin Hypercube) for UQ/surrogate training.

    Args:
        uncertain_parameters (dict): dictionary of original defined uncertain parameters.

    Returns:
        dict: a dictionary containing all generated samples.
    """
    sample_size = uncertain_parameters["sample_size"]
    sampling_method = uncertain_parameters["sampling_method"]
    seed = uncertain_parameters["seed"]
    # Create the shared RandomState object for sequential draws (avoids rank correlation)
    rng = np.random.RandomState(seed) if seed is not None else None

    sample_param_list = []
    
    if "geometry" in uncertain_parameters["uncertain_parameters"]:
        sample_param_list.append(geometry_dist(**uncertain_parameters))
    if "rock_data" in uncertain_parameters["uncertain_parameters"]:
        sample_param_list.append(medium_properties_dist(**uncertain_parameters))
    if "nuclide_sorption_data" in uncertain_parameters["uncertain_parameters"]:
        sample_param_list.append(nuclide_properties_dist(nuclide_prop = "nuclide_sorption_data", **uncertain_parameters))
    if "nuclide_water_diffusivity_data" in uncertain_parameters["uncertain_parameters"]:
        sample_param_list.append(nuclide_properties_dist(nuclide_prop = "nuclide_water_diffusivity_data", **uncertain_parameters))

    if len(sample_param_list) == 0:
        raise ValueError("No uncertain parameters defined to sample.")
    
    param_list = [item for sublist in sample_param_list for item in sublist]

    if sampling_method in ['sobol', 'latin_hypercube']:
        
        ndim = len(param_list)
            
        # 2. Draw uniform QMC samples in [0, 1)^ndim
        if sampling_method == 'sobol':
            sampler = qmc.Sobol(d=ndim, scramble=True, seed=seed)
        else: # latin_hypercube
            sampler = qmc.LatinHypercube(d=ndim, scramble=True, seed=seed)
            
        qmc_raw = sampler.random(n=sample_size) # shape (sample_size, ndim)
        
        # 3. Apply inverse CDF (PPF) of each target distribution to transform the samples
        samples_transformed = np.zeros_like(qmc_raw)
        for j, param in enumerate(param_list):
            samples_transformed[:, j] = param['dist'].ppf(qmc_raw[:, j])
            
    # 4. Unpack the flat samples array back into the expected hierarchical dict structure
    
    temp_unit = {} 
    temp_medium = {}
    temp_nuclide = {}
    
    for j, param in enumerate(param_list):
        category = param['category']
        layer = param['layer']
        key = param['prop']

        if sampling_method in ['sobol', 'latin_hypercube']:
            col_samples = samples_transformed[:, j]
        elif sampling_method == 'random':
            col_samples = param['dist'].rvs(size=sample_size, random_state=seed)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")
        
        if category == 'geometry':
            if key not in temp_unit:
                temp_unit[key] = {}
            temp_unit[key][layer] = col_samples
        elif category == 'rock_data':
            if layer not in temp_medium:
                temp_medium[layer] = {}
            temp_medium[layer][key] = col_samples
        elif category == 'nuclide_properties':
            nucl, prop_name = key
        
            if prop_name not in temp_nuclide:
                temp_nuclide[prop_name] = {}
            if layer not in temp_nuclide[prop_name]:
                temp_nuclide[prop_name][layer] = {}
            temp_nuclide[prop_name][layer].update({nucl: col_samples})
            
    # Structure the material properties 
    sampled_parameters = {'uncertain_parameters': {}}
    if temp_unit:
        sampled_parameters['uncertain_parameters']['geometry'] = temp_unit
    if temp_medium:
        sampled_parameters['uncertain_parameters']["rock_data"] = temp_medium
    if temp_nuclide:
       sampled_parameters['uncertain_parameters'].update(temp_nuclide)
   
    return sampled_parameters


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

NoAliasDumper.add_representer(
    np.ndarray,
    lambda dumper, data: dumper.represent_list(data.tolist())
)

def save_sampled_data(sampled_parameters, path_to_save, file_type):
    """Function to save numpy.ndarray object to specified file type.

    Args:
        sampled_parameters (dict): a dictionary containing all generated samples.
        path_to_save (str): path to the output file.
        file_type (str): file type to save ('YAML', 'pickle', or 'HDF5').
    """
    if file_type.lower() == "yaml":
        with open(path_to_save+".yaml", "w") as f:
            yaml.dump(
                sampled_parameters,
                f,
                Dumper=NoAliasDumper,
                sort_keys=False,
            )
    elif file_type.lower() == "pickle":
        with gzip.open(path_to_save+'.pkl.gz', "wb") as f:
            pickle.dump(sampled_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif file_type.lower() == "hdf5":
        save_sampled_data_to_hdf5(sampled_parameters, path_to_save+'.h5')
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Expected one of: 'YAML', 'pickle', 'HDF5'.")
    

def save_sampled_data_to_hdf5(sampled_parameters, path_to_save_hdf5):
    """Function to save numpy.ndarray object to HDF5 file.

    Args:
        sampled_parameters (dict): a dictionary containing all generated samples.
        path_to_save_hdf5 (str): path to the HDF5 file.
    """

    def save(group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                save(subgroup, value)
            else:
                group.create_dataset(key, data=np.asarray(value))

    with h5py.File(path_to_save_hdf5, "w") as f:
        save(f, sampled_parameters)

def load_dict_from_hdf5(file_path) -> dict:
    def load_group(group):
        result = {}

        # Load datasets
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[:]

            elif isinstance(item, h5py.Group):
                result[key] = load_group(item)

        # Load attributes
        for key, value in group.attrs.items():
            result[key] = value

        return result

    with h5py.File(file_path, "r") as f:
        return load_group(f)

def load_sampled_data(file_path) -> dict:
    file_path = Path(file_path)
    suffix = file_path.suffixes
    # YAML
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    # Pickle gzip
    elif suffix[-2:] == [".pkl", ".gz"]:
        with gzip.open(file_path, "rb") as f:
            return pickle.load(f)
    # HDF5
    elif file_path.suffix.lower() in {".h5", ".hdf5"}:
        return load_dict_from_hdf5(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. "
            "Expected .yaml, .yml, .pkl.gz, .h5, or .hdf5."
        )

def parse_args():

    parser = argparse.ArgumentParser(
        description="Create sampling data based on a YAML configuration file.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the sample configuration YAML file.",
    )

    parser.add_argument(
            "--sample_size",
            type=str,
            required=True,
            help="Number of samples to generate.",
        )

    parser.add_argument(
                "--sampling_method",
                choices=["random", "sobol", "latin_hypercube"],
                default="random",
                help="random for SciPy-based sampling, or sobol / latin_hypercube for SciPy.",
            )

    parser.add_argument(
                "--seed",
                type=str,
                required=True,
                help="random seed for reproducibility.",
            )

    parser.add_argument(
            "--path_to_geometry_data",
            type=str,
            required=False,
            help="Path to the geometry data.",
        )

    parser.add_argument(
        "--path_to_rock_data",
        type=str,
        required=False,
        help="Path to the rock data.",
    )

    parser.add_argument(
        "--path_to_sorption_data",
        type=str,
        required=False,
        help="Path to the sorption data",
    )

    parser.add_argument(
            "--path_to_diffusivity_data",
            type=str,
            required=False,
            help="Path to the nuclide diffusivity data",
        )

    parser.add_argument(
        "--path_to_save_sampled_data",
        type=str,
        required=True,
        help="Output directory for sampled data.",
    )

    parser.add_argument(
    "--save_file_type",
    choices=["YAML", "pickle", "HDF5"],
    default="HDF5",
    help="File types to be saved; defaults to HDF5.",
)

    return parser.parse_args()

def load_yaml_config(config_path):
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the input YAML configuration file

    Raises:
        FileNotFoundError: not found message.
        ValueError: empty config file message.

    Returns:
        dict: configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        msg = f"Config file is empty: {config_path}"
        raise ValueError(msg)

    return config


def build_sample_config(config_path, sample_size, sampling_method, seed, path_to_geometry_data, path_to_rock_data, path_to_sorption_data, path_to_diffusivity_data):
    """Load and validate, a site configuration file. Save sampled data with output path given via CLI.

    Args:
        config_path (str): path to the configuration file.
        sample_size (int): Number of samples to generate.
        sampling_method (str): Sampling method to use ('random', 'sobol', or 'latin_hypercube').
        seed (int): Random seed for reproducibility.
        path_to_geometry_data (str): Path to the geometry data.
        path_to_rock_data (str): Path to the rock data.
        path_to_sorption_data (str): Path to sorption coefficient data.
        path_to_diffusivity_data (str): Path to nuclide diffusivity data.

    Returns:
        dict: configuration dictionary with output paths given via CLI.
    """
    raw_config = load_yaml_config(config_path)
    
    raw_config["sample_size"] = int(sample_size)
    raw_config["sampling_method"] = sampling_method
    raw_config["seed"] = int(seed)
    raw_config["path_to_geometry_data"] = path_to_geometry_data
    raw_config["path_to_rock_data"] = path_to_rock_data
    raw_config["path_to_sorption_data"] = path_to_sorption_data
    raw_config["path_to_diffusivity_data"] = path_to_diffusivity_data
    

    return raw_config

def main() -> None:
    args = parse_args()

    try:
        sample_config = build_sample_config(args.config, args.sample_size, args.sampling_method, args.seed, args.path_to_geometry_data, args.path_to_rock_data, args.path_to_sorption_data, args.path_to_diffusivity_data)
    except (FileNotFoundError, ValueError):
        sys.exit(1)

    
    Path(args.path_to_save_sampled_data).mkdir(parents=True, exist_ok=True)

    sampled_data = sample_parameters(sample_config)

    save_type = args.save_file_type.lower()
    path_to_save_sampled_data_file = os.path.join(args.path_to_save_sampled_data, f"sample_{sample_config['sampling_method']}_size{sample_config['sample_size']}_seed{sample_config['seed']}")
    save_sampled_data(sampled_data, path_to_save=path_to_save_sampled_data_file, file_type=save_type)

if __name__ == "__main__":
    main()