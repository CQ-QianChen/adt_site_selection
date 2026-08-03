#!/bin/bash
python -m smart_data_hub.export_data \
--config ./communication_sdh/input/config_Case1_Claystone_Baden-Wuerttemberg.yaml \
--path_to_save_rock_yaml output/Case1_Claystone_Baden-Wuerttemberg/rock_data \
--path_to_save_site_yaml output/Case1_Claystone_Baden-Wuerttemberg/site_data \
--path_to_save_site_geometry output/Case1_Claystone_Baden-Wuerttemberg/geometry