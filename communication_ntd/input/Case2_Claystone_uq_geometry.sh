#!/bin/bash
python -m nuctransportdb.export_data \
--config ./communication_ntd/input/config_Case2_Claystone_uq_geometry.yaml \
--path_to_site_yaml_file output/Case2_Claystone_uq_geometry/site_data/DE_South_Claystone.yaml \
--path_to_save_sorption_data output/Case2_Claystone_uq_geometry/nuclide_data/sorption_data \
--path_to_save_nuclide_species_data output/Case2_Claystone_uq_geometry/nuclide_data \
--path_to_save_nuclide_emitted_energy_data output/Case2_Claystone_uq_geometry/nuclide_data \