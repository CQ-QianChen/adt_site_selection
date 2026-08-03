#!/bin/bash
python -m smart_data_hub.export_data \
--config ./communication_sdh/input/config_Case2_Claystone_uq_geometry.yaml \
--path_to_save_rock_yaml output/Case2_Claystone_uq_geometry/rock_data \
--path_to_save_site_yaml output/Case2_Claystone_uq_geometry/site_data \
--path_to_save_site_geometry output/Case2_Claystone_uq_geometry/geometry