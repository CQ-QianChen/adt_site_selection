#!/bin/bash
python -m nuctransportdb.export_data \
--config ./communication_ntd/input/config_Case1_Claystone_Baden-Wuerttemberg.yaml \
--path_to_site_yaml_file output/Case1_Claystone_Baden-Wuerttemberg/site_data/DE_South_Claystone.yaml \
--path_to_save_sorption_data output/Case1_Claystone_Baden-Wuerttemberg/sorption_data \
--path_to_save_nuclide_species_data output/Case1_Claystone_Baden-Wuerttemberg/nuclide_data \
--path_to_save_nuclide_emitted_energy_data output/Case1_Claystone_Baden-Wuerttemberg/emitted_energy_data \