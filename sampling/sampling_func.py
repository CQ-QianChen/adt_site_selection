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
    if kwargs.get('units') is not None:
        for layer, bounds in kwargs.get('units').items():
            # Use scipy.stats.uniform to draw samples. loc is lower bound, scale is width of the interval.
            geometry_uncertainties.append({
                    'category': 'units',
                    'layer': layer,
                    'prop': 'geometry',
                    'dist': scipy.stats.uniform(loc=bounds[0], scale=bounds[1] - bounds[0])
                })
    
        return geometry_uncertainties
    else:
        raise ValueError("No geometry uncertainties specified.")

def medium_properties_dist(**kwargs):
    """
    Function to generate sample distribution for medium properties.

    Args:
        **kwargs: additional keyword arguments for medium properties uncertainties. 
            medium_properties (dict): dictionary containing rock units and their corresponding uncertain properties.

    Returns:
        list(dict): a list of dictionaries containing sample distribution medium properties.
    """
    medium_props_samples = []
    if kwargs.get("medium_properties") is not None:
        for rock_unit, uncertain_rock_properties in kwargs["medium_properties"].items():
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
                            'category': 'medium_properties',
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

def nuclide_sorption_properties_dist(**kwargs):
    """
    Function to generate sample distribution for nuclide properties.

    Args:
        **kwargs: additional keyword arguments for nuclide properties uncertainties. 
            nuclide_properties (dict): dictionary containing rock units, nuclides, and their corresponding uncertain properties.
    Returns:
        list(dict): a list of dictionaries containing sample distribution nuclide properties.
    """
    nuclides_props_samples = []

    if kwargs.get("nuclide_properties") is not None:
        for rock_unit, uncertain_nuclide_properties in kwargs["nuclide_properties"].items():
            
            for nuclide, nuclide_properties in uncertain_nuclide_properties.items():
                
                nuclide_file = os.path.join(kwargs.get("path_to_sorption_data"), rock_unit+".yaml")
                with open(nuclide_file, "r", encoding="utf-8") as f:
                    nuclide_data = yaml.safe_load(f)

                # Get samples for each property from the PDF string
                for np_prop in nuclide_properties:
                    try:
                        # Locate the specific nuclide item in the list for the given property
                        np_rvs = nuclide_data[nuclide][0]["probability_distribution"]["sampled_data"]
                        if np_rvs:
                            nuclides_props_samples.append({
                                'category': 'nuclide_properties',
                                'layer': rock_unit,
                                'prop': (nuclide, np_prop),
                                'dist': eval(np_rvs[:np_rvs.index(".rvs(")])
                            })
                        else:  # np_rvs is None
                            raise ValueError(
                                f"No probability distribution was provided for property '{np_prop}' "
                                f"of nuclide '{nuclide}' in rock unit '{rock_unit}'. "
                                f"This property is treated as constant with its specified deterministic value {nuclide_data[nuclide][0]['value']}. "
                                f"If you want to consider uncertainties for this property, you must provide "
                                f"a probability distribution explicitly in file {nuclide_file}."
                            )

                    except KeyError:
                        raise KeyError(f"Property {np_prop} for nuclide {nuclide} not found for rock unit {rock_unit}. The available properties are: {list(nuclide_data.keys())}")
                        
        return nuclides_props_samples
    else:
        raise ValueError("No uncertainties of nuclide properties specified.")

def combine_sampled_material_parameters(**kwargs):
    """
    Function to combine all sampled material parameters into a single dictionary.

    Args:
        **kwargs: additional keyword arguments containing sampled medium properties and/or nuclide properties.

    Returns:
        dict: a combined dictionary containing all sampled parameters grouped by rock unit.

    """
    phase_properties = {
        rock_unit: {
            f"{phase_type}_{phase_prop}": values
            for phase_type, phase_props in phase_props_dict.items()
            for phase_prop, values in phase_props.items()
        }
        for rock_unit, phase_props_dict in kwargs.get("phase_properties", {}).items()
    }

    nuclide_properties = {
        rock_unit: {
            f"{nuclide}_{nuclide_prop}": values
            for nuclide, nuclide_props in nuclide_props_dict.items()
            for nuclide_prop, values in nuclide_props.items()
        }
        for rock_unit, nuclide_props_dict in kwargs.get("nuclide_properties", {}).items()
    }

    # Use dict.fromkeys to merge keys while preserving insertion order (geological/configuration order)
    all_rock_units = list(dict.fromkeys(
        list(kwargs.get("medium_properties", {}).keys()) +
        list(nuclide_properties.keys()) +
        list(phase_properties.keys())
    ))

    sampled_material_parameters_combined = {
        rock_unit: {
            # Unpack medium properties first, then nuclide, then phase to match configuration ordering
            **kwargs.get("medium_properties", {}).get(rock_unit, {}),
            **nuclide_properties.get(rock_unit, {}),
            **phase_properties.get(rock_unit, {})
        }
        for rock_unit in all_rock_units
    }

    return sampled_material_parameters_combined

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
    
    if "units" in uncertain_parameters:
        sample_param_list.append(geometry_dist(**uncertain_parameters))
    if "medium_properties" in uncertain_parameters:
        sample_param_list.append(medium_properties_dist(**uncertain_parameters))
    if "nuclide_properties" in uncertain_parameters:
        sample_param_list.append(nuclide_sorption_properties_dist(**uncertain_parameters))

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
    sampled_parameters = {}
    if "units" in uncertain_parameters:
        sampled_parameters["units"] = {}
        
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
        
        if category == 'units':
            sampled_parameters["units"][layer] = col_samples
        elif category == 'medium_properties':
            if layer not in temp_medium:
                temp_medium[layer] = {}
            temp_medium[layer][key] = col_samples
        elif category == 'nuclide_properties':
            nucl, prop_name = key
            if layer not in temp_nuclide:
                temp_nuclide[layer] = {}
            if nucl not in temp_nuclide[layer]:
                temp_nuclide[layer][nucl] = {}
            temp_nuclide[layer][nucl][prop_name] = col_samples
            
    # Structure the material properties using combine_sampled_material_parameters
    material_kwargs = {}
    if temp_medium:
        material_kwargs["medium_properties"] = temp_medium
        sampled_parameters["medium_properties"] = temp_medium
    if temp_nuclide:
        material_kwargs["nuclide_properties"] = temp_nuclide
        sampled_parameters["nuclide_properties"] = temp_nuclide

    if material_kwargs:
        sampled_parameters["material_properties"] = combine_sampled_material_parameters(**material_kwargs)
        
    return sampled_parameters

def save_sampled_data_to_pickle(sampled_parameters, path_to_save_pickle):
    """Function to save numpy.ndarray object to pickle file.

    Args:
        sampled_parameters (dict): a dictionary containing all generated samples.
        path_to_save_pickle (str): path to the pickle file.
    """


    with gzip.open(path_to_save_pickle, "wb") as f:
        pickle.dump(sampled_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

NoAliasDumper.add_representer(
    np.ndarray,
    lambda dumper, data: dumper.represent_list(data.tolist())
)

def save_sampled_data_to_yaml(sampled_parameters, path_to_save_yaml):
    """Function to save numpy.ndarray object to YAML file.

    Args:
        sampled_parameters (dict): a dictionary containing all generated samples.
        path_to_save_yaml (str): path to the YAML file.
    """

    with open(path_to_save_yaml, "w") as f:
        yaml.dump(
            sampled_parameters,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
        )


REQUIRED_FIELDS = ["sampling_method", "seed"]

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
        "--path_to_save_sampled_data",
        type=str,
        required=True,
        help="Output directory for sampled data.",
    )

    parser.add_argument(
    "--save_file_type",
    choices=["YAML", "pickle"],
    default="pickle",
    help="File types to be saved; defaults to pickle.",
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

def validate_config(config) -> None:
    """Validate that all required fields are present in the config.

    Args:
        config (dict): configuration dictionary

    Raises:
        ValueError: missing required field message.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        msg = f"Missing required config field(s): {missing}"
        raise ValueError(msg)

def build_sample_config(config_path, path_to_rock_data, path_to_sorption_data):
    """Load and validate, a site configuration file. Save sampled data with output path given via CLI.

    Args:
        config_path (str): path to the configuration file.
        path_to_rock_data (str): Path to the rock data.
        path_to_sorption_data (str): Path to sorption coefficient data.

    Returns:
        dict: configuration dictionary with output paths given via CLI.
    """
    raw_config = load_yaml_config(config_path)
    validate_config(raw_config)

    raw_config["path_to_rock_data"] = path_to_rock_data
    raw_config["path_to_sorption_data"] = path_to_sorption_data

    return raw_config

def main() -> None:
    args = parse_args()

    try:
        sample_config = build_sample_config(args.config, args.path_to_rock_data, args.path_to_sorption_data)
    except (FileNotFoundError, ValueError):
        sys.exit(1)

    
    Path(args.path_to_save_sampled_data).mkdir(parents=True, exist_ok=True)

    sampled_data = sample_parameters(sample_config)

    save_type = args.save_file_type.lower()

    if save_type == "yaml":
        path_to_save_yaml = os.path.join(args.path_to_save_sampled_data, f"sample_{sample_config['sampling_method']}_size{sample_config['sample_size']}_seed{sample_config['seed']}.yaml")
        save_sampled_data_to_yaml(sampled_data, path_to_save_yaml)
    elif save_type == "pickle":
        path_to_save_pickle = os.path.join(args.path_to_save_sampled_data, f"sample_{sample_config['sampling_method']}_size{sample_config['sample_size']}_seed{sample_config['seed']}.pkl.gz")
        save_sampled_data_to_pickle(sampled_data, path_to_save_pickle)
    else:
        ValueError(
        f"Unsupported save_file_type: {args.save_file_type!r}. "
        "Expected one of: 'YAML', 'pickle'.")

if __name__ == "__main__":
    main()