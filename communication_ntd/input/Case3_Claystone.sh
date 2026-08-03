#!/bin/bash
python -m nuctransportdb.export_data \
--config ./communication_ntd/input/config_Case3_Claystone.yaml \
--path_to_site_yaml_file output/Case3_Claystone/site_data/DE_South_Claystone.yaml \
--path_to_save_sorption_data output/Case3_Claystone/sorption_data \
--path_to_save_nuclide_species_data output/Case3_Claystone/nuclide_data \
--path_to_save_nuclide_emitted_energy_data output/Case3_Claystone/emitted_energy_data \