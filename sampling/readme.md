# sampling

Draws (or applies fixed) sample values for material properties and geometry,
without running any OGS simulations

## Usage (example)

```
#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case2_Claystone_uq_geometry.yaml \
--path_to_rock_data ../output/Case2_Claystone_uq_geometry/rock_data \
--path_to_sorption_data ../output/Case2_Claystone_uq_geometry/sorption_data \
--path_to_save_sampled_data ../output/Case2_Claystone_uq_geometry/sampled_data \
--save_file_type pickle \
```

## input (a config YAML file)

```yaml
- `sample_size`: number of samples to draw.
- `sample_type`: `random` for SciPy-based sampling, or`sobol` / `latin_hypercube` for SciPy QMC experimental design.
- `seed`: fixes the draws so the same seed always reproduces the same
output.
- `units`: `{layer_name: [low, high]}` - the range each layer's bottom
  depth is drawn from.
- `medium_properties`: `{rock: [prop_name_or_{prop: [values]}, ...]}` - a
  bare property name draws it from the database's distribution; a
  `{prop: [values]}` entry uses exactly those fixed values instead (must
  list `sample_size` of them).
- `nuclide_properties`: `{rock: {nuclide: [same shape as medium_properties'
  list]}}`.
```
- Note: 
  - Entries for `units`, `medium_properties` and `nuclide_properties` are optional.
  - `units` are drawn uniformly at random within the given
    `[low, high]` range, and every other property is drawn from whatever
    distribution is defined for it in the database 

## output (YAML or pickle file) 
```yaml
- `units`:  `{layer_name: [values]}`
- `medium_properties`: `{rock: {property_name: [values]}}`
- `nuclide_properties`: `{rock: {nuclide: {property_name: [values]}}}`
- `material_properties`: `{rock: {property_name: [values]}}` - a
  nuclide-specific property is named `"{nuclide}_{property}"`.
```
- Note: `material_properties` combines outputs from `medium_properties` and `material_properties`