#!/bin/bash
python -m sampling_func \
--config ./input/sampling_config_Case3_Rocksalt.yaml \
--path_to_rock_data ../output/Case3_Rocksalt/rock_data \
--path_to_sorption_data ../output/Case3_Rocksalt/nuclide_data/sorption_data \
--path_to_save_sampled_data ../output/Case3_Rocksalt/sampled_data \
--save_file_type YAML \