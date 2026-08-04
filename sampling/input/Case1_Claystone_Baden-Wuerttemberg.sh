#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case1_Claystone_Baden-Wuerttemberg.yaml \
--path_to_rock_data ../output/Case1_Claystone_Baden-Wuerttemberg/rock_data \
--path_to_sorption_data ../output/Case1_Claystone_Baden-Wuerttemberg/nuclide_data/sorption_data \
--path_to_save_sampled_data ../output/Case1_Claystone_Baden-Wuerttemberg/sampled_data \
--save_file_type YAML \