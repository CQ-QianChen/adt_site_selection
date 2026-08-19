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