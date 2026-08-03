#!/bin/bash
python -m smart_data_hub.export_data \
--config ./communication_sdh/input/config_Case3_Crystalline.yaml \
--path_to_save_rock_yaml output/Case3_Crystalline/rock_data \
--path_to_save_site_yaml output/Case3_Crystalline/site_data \
--path_to_save_site_geometry output/Case3_Crystalline/geometry