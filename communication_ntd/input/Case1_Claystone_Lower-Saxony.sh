#!/bin/bash
python -m nuctransportdb.export_data \
--config ./communication_ntd/input/config_Case1_Claystone_Lower-Saxony.yaml \
--path_to_site_yaml_file output/Case1_Claystone_Lower-Saxony/site_data/DE_North_Claystone.yaml \
--path_to_save_sorption_data output/Case1_Claystone_Lower-Saxony/nuclide_sorption_data \
--path_to_save_nuclide_species_data output/Case1_Claystone_Lower-Saxony/nuclide_species_data \
--path_to_save_nuclide_water_diffusivity_data output/Case1_Claystone_Lower-Saxony/nuclide_water_diffusivity_data \
--path_to_save_nuclide_emitted_energy_data output/Case1_Claystone_Lower-Saxony/nuclide_emitted_energy_data \