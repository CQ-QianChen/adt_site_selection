#!/bin/bash
python -m nuctransportdb.export_data \
--config ./communication_ntd/input/config_Case3_Crystalline.yaml \
--path_to_site_yaml_file output/Case3_Crystalline/site_data/DE_Crystalline.yaml \
--path_to_save_sorption_data output/Case3_Crystalline/sorption_data \
--path_to_save_nuclide_species_data output/Case3_Crystalline/nuclide_data \
--path_to_save_nuclide_emitted_energy_data output/Case3_Crystalline/emitted_energy_data \