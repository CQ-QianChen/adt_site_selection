#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case3_Claystone.yaml \
--path_to_rock_data ../output/Case3_Claystone/rock_data \
--path_to_sorption_data ../output/Case3_Claystone/sorption_data \
--path_to_save_sampled_data ../output/Case3_Claystone/sampled_data \
--save_file_type pickle \