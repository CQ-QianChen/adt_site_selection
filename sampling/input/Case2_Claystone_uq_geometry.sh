#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case2_Claystone_uq_geometry.yaml \
--path_to_rock_data ../output/Case2_Claystone_uq_geometry/rock_data \
--path_to_sorption_data ../output/Case2_Claystone_uq_geometry/sorption_data \
--path_to_save_sampled_data ../output/Case2_Claystone_uq_geometry/sampled_data \
--save_file_type pickle \