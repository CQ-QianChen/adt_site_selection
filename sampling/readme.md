# sampling

Draws (or applies fixed) sample values for material properties and geometry,
without running any OGS simulations

## Usage (example)

Go to the directory that contains ``sampling_func.py``, then run:


optional (allow the access to the file):
```console
chmod +x ../input/Case3_Claystone.sh
```
Then:
```console
./input/Case3_Claystone.sh
```

The ``Case3_Claystone.sh`` file has the following entries:

```
#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case3_Claystone.yaml \
--sample_size 4 \
--sampling_method sobol \
--seed 21 \
--path_to_geometry_data ../output/Case3_Claystone/geometry \
--path_to_rock_data ../output/Case3_Claystone/rock_data \
--path_to_sorption_data ../output/Case3_Claystone/nuclide_sorption_data \
--path_to_diffusivity_data ../output/Case3_Claystone/nuclide_water_diffusivity_data \
--path_to_save_sampled_data ../output/Case3_Claystone/sampled_data \
--save_file_type HDF5 \
```

- Note: use "random", "sobol",or  "latin_hypercube" for the `--sampling_method` argument. The default is "random".

## input (a config YAML file)

```yaml
uncertain_parameters:
  geometry:
    rock_interface:
      - interface_name: 
      -...
  rock_data:
    rock_name1:
      - rock_property_name1
      - ...
    rock_name2:
    ...

  nuclide_water_diffusivity_data:
    rock_name1:
      - nuclide1
      - ...
    rock_name2:
    ...
  nuclide_sorption_data:
    rock_name1:
      - nuclide1
      - ...
    rock_name2:
    ...
```
- Note: 
  - Entries for `geometry`, `nuclide_water_diffusivity_data` and `nuclide_sorption_data` are optional.

## output (YAML, pickle, or HDF5 file) 
```yaml
uncertain_parameters:
  geometry:
    rock_interface:
      interface_name: [...]
      ...
  rock_data:
    rock_name1:
      rock_property_name1: [...]
      ...
    rock_name2:
    ...

  nuclide_water_diffusivity_data:
    rock_name1:
      nuclide1: [...]
      ...
    rock_name2:
    ...
  nuclide_sorption_data:
    rock_name1:
      nuclide1: [...]
      ...
    rock_name2:
    ...
```
- Note: 
  - Sampled values for appended to the lists.

### Read saved sampled data back to Python dictionary, use load_sampled_data function from sampling_func.py, e.g.:

```python
from sampling_func import load_sampled_data

sampled_data_file_path = "../..h5" # make sure you have the right path
sampled_data = load_sampled_data(sampled_data_file_path)
```