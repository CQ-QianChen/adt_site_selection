#!/bin/bash
python -m nuctransportdb.export_data \
--config ./communication_ntd/input/config_Case3_Rocksalt.yaml \
--path_to_site_yaml_file output/Case3_Rocksalt/site_data/DE_Rocksalt.yaml \
--path_to_save_sorption_data output/Case3_Rocksalt/nuclide_sorption_data \
--path_to_save_nuclide_species_data output/Case3_Rocksalt/nuclide_species_data \
--path_to_save_nuclide_water_diffusivity_data output/Case3_Rocksalt/nuclide_water_diffusivity_data \
--path_to_save_nuclide_emitted_energy_data output/Case3_Rocksalt/nuclide_emitted_energy_data \