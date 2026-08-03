#!/bin/bash
python -m smart_data_hub.export_data \
--config ./communication_sdh/input/config_Case3_Rocksalt.yaml \
--path_to_save_rock_yaml output/Case3_Rocksalt/rock_data \
--path_to_save_site_yaml output/Case3_Rocksalt/site_data \
--path_to_save_site_geometry output/Case3_Rocksalt/geometry