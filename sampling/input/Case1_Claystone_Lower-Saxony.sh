#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case1_Claystone_Lower-Saxony.yaml \
--sample_size 4 \
--sampling_method sobol \
--seed 21 \
--path_to_geometry_data ../output/Case1_Claystone_Lower-Saxony/geometry \
--path_to_rock_data ../output/Case1_Claystone_Lower-Saxony/rock_data \
--path_to_sorption_data ../output/Case1_Claystone_Lower-Saxony/nuclide_sorption_data \
--path_to_diffusivity_data ../output/Case1_Claystone_Lower-Saxony/nuclide_water_diffusivity_data \
--path_to_save_sampled_data ../output/Case1_Claystone_Lower-Saxony/sampled_data \
--save_file_type HDF5 \