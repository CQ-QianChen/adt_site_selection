#!/bin/bash
python -m smart_data_hub.export_data \
--config ./communication_sdh/input/config_Case3_Claystone.yaml \
--path_to_save_rock_yaml output/Case3_Claystone/rock_data \
--path_to_save_site_yaml output/Case3_Claystone/site_data \
--path_to_save_site_geometry output/Case3_Claystone/geometry