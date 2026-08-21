## Usage

### Installation
create a conda environment using ``environment.yml``. Go to the ``model`` directory and run the following command ``conda env create -f environment.yml``, followed by ``conda activate yaml2nuctrans``.

### Example

To run the simluation, use ``yaml2nuctrans.get_model.ogs_model``. This script reads a YAML configuration file as well as data files for geometry, rock properties, sorption data, and diffusivity data. It then generates OGS models and save the results into a HDF5 file. For example:

#### Run in the command line (example)

Go to the OGS_1D_SIM directory, in the ``yaml2nuctrans`` conda environment, run 

optional (allow the access to the file):
```console
chmod +x ./model/input/Case3_Claystone.sh
```
Then:
```console
./model/input/Case3_Claystone.sh
```

The bash file structures as follows:

```commandline
python -m yaml2nuctrans.get_model.ogs_model \
    --output_directory ./output/Case3_Claystone/model_results/ \
    --rock_data_folder_path ./output/Case3_Claystone/rock_data \
    --site_folder_path ./output/Case3_Claystone/site_data \
    --geometry_folder_path ./output/Case3_Claystone/geometry \
    --emitted_energy_folder_path ./output/Case3_Claystone/nuclide_emitted_energy_data \
    --species_type_folder_path ./output/Case3_Claystone/nuclide_species_data \
    --sorption_data_folder_path ./output/Case3_Claystone/nuclide_sorption_data \
    --nuclide_water_diffusivity_folder_path ./output/Case3_Claystone/nuclide_water_diffusivity_data \
    --model_config_path ./model/input/config_Case3_Claystone.yaml \
    --sampled_data_file_path ./output/Case3_Claystone/sampled_data/sample_sobol_size4_seed21.h5 \
    --get_field_component_index "[0, 1, 2]" \
    --sort_by_index 2 \
    --run_mode ensemble \
    --parallel True \
    --n_jobs 4 \
    --keep_vtu False \
    --path_to_save_results_hdf5_file default \
    --save_sampled_data True \

```


The input arguments are as follows:

| Argument                                | Type      | Required      | Default   | Example value                                                           | Description                                                                        |   |
|-----------------------------------------|-----------|---------------|-----------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------|---|
| --output_directory                      | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/model_results/                            | Directory where per-sample model outputs are written                               |   |
| --rock_data_folder_path                 | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/rock_data                                 | Folder with rock property data (density, porosity, …)                              |   |
| --site_folder_path                      | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/site_data                                 | Folder with site-specific data                                                     |   |
| --geometry_folder_path                  | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/geometry                                  | Folder with geometry data (rock interfaces)                                        |   |
| --emitted_energy_folder_path            | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/nuclide_emitted_energy_data               | Folder with nuclide emitted energy data                                            |   |
| --species_type_folder_path              | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/nuclide_species_data                      | Folder with nuclide species type data                                              |   |
| --sorption_data_folder_path             | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/nuclide_sorption_data                     | Folder with nuclide sorption coefficient data                                      |   |
| --nuclide_water_diffusivity_folder_path | path      | Yes           | —         | ./Case2_Claystone_uq_geometry/nuclide_water_diffusivity_data            | Folder with nuclide water diffusivity data                                         |   |
| --model_config_path                     | path      | Yes           | —         | ./input/config_Case2_Claystone_uq_geometry.yaml                         | YAML simulation config file                                                        |   |
| --sampled_data_file_path                | path      | ensemble only | None      | ./Case2_Claystone_uq_geometry/sampled_data/sample_sobol_size4_seed21.h5 | HDF5 file with sampled uncertain parameters                                        |   |
| --get_field_component_index             | list[int] | No            | [0, 1, 2] | "[0, 1, 2]"                                                             | Which vector components of the fields to extract (e.g. [2] or [0, 2] or [0, 1, 2]. |   |
| --sort_by_index                         | int       | No            | 2         | 2                                                                       | Coordinate index used to sort the results (0=x, 1=y, 2=z)                          |   |
| --run_mode                              | str       | No            | ensemble  | single                                                                  | single (best estimate) or ensemble (UQ run)                                        |   |
| --parallel                              | bool      | No            | True      | True                                                                    | Run ensemble samples in parallel via joblib                                        |   |
| --n_jobs                                | int       | No            | -1        | 4                                                                       | Number of parallel worker processes (-1 = all CPUs)                                |   |
| --keep_vtu                              | bool      | No            | False     | False                                                                   | Keep per-sample VTU/PVD output folders after results are merged                    |   |
| --path_to_save_results_hdf5_file        | path      | No            | default   | default                                                                 | Where to write the results HDF5; default = inside output_directory                 |   |
| --save_sampled_data                     | bool      | ensemble only | True      | True                                                                    | Include sampled input parameters in the results HDF5 file                          |   |

#### Note:
##### get_field_component_index examples: 
  
| get_field_component_index | key example  | keys shape  | I-129 values shape  | concentration values example               | I-129Flux values shape | Flux values example                                                                 |
|---------------------------|--------------|-------------|---------------------|--------------------------------------------|------------------------|-------------------------------------------------------------------------------------|
| [2]                       | (z, t)       | (n_keys, 2) | (n_keys, n_samples) | {(k2, t): [s0, s1, None, s3], ...},        | (n_keys, n_samples, 1) | {(k2, t): [s0, s1, None, s3], ...}}                                                 |
| [0, 2]                    | (x, z, t)    | (n_keys, 3) | (n_keys, n_samples) | {(k0, k2, t): [s0, s1, None, s3], ...}     | (n_keys, n_samples, 2) | {(k0, k2, t): [[s0_0, s0_2], None, [...], ..., [...], [sn_0, sn_2]]}                |
| [0, 1, 2]                 | (x, y, z, t) | (n_keys, 4) | (n_keys, n_samples) | {(k0, k1, k2, t): [s0, s1, None, s3], ...} | (n_keys, n_samples, 3) | {(k0, k1, k2, t): [[s0_0, s0_1, s0_2], None, [...], ..., [...], [sn_0,sn_1, sn_2]]} |

##### path_to_save_results_hdf5_file:
- If not specified, the results HDF5 file will be saved in the output_directory, with the name:
  - f"ensemble_results_with_{Path(sampled_data_file_path).stem}.h5" for ensemble run and containing sampled data, or
  - f"ensemble_results_using_{Path(sampled_data_file_path).stem}.h5" for ensemble run using sampled data but not saving sampled data, or
  - f"single_best_estimate_results.h5" for single run using the best estimate values of the uncertain parameters.
- If specified, the results HDF5 file will be saved in the specified path.

### Read saved simulation data back to Python dictionary, use load_results_from_hdf5 function from yaml2nuctrans.read_results.read_ogs_model_results e.g.:

```python
from yaml2nuctrans.read_results.read_ogs_model_results import load_results_from_hdf5

results_file_path = "../...h5"  # make sure you have the right path

loaded_results = load_results_from_hdf5(results_file_path)

sampled_data = loaded_results.get('sampled_data')
sim_results = loaded_results.get('simulation_results')
```