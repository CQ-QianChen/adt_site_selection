
python -m yaml2nuctrans.get_model.ogs_model \
    --output_directory ./output/Case3_Rocksalt/model_results/ \
    --rock_data_folder_path ./output/Case3_Rocksalt/rock_data \
    --site_folder_path ./output/Case3_Rocksalt/site_data \
    --geometry_folder_path ./output/Case3_Rocksalt/geometry \
    --emitted_energy_folder_path ./output/Case3_Rocksalt/nuclide_emitted_energy_data \
    --species_type_folder_path ./output/Case3_Rocksalt/nuclide_species_data \
    --sorption_data_folder_path ./output/Case3_Rocksalt/nuclide_sorption_data \
    --nuclide_water_diffusivity_folder_path ./output/Case3_Rocksalt/nuclide_water_diffusivity_data \
    --model_config_path ./model/input/config_Case3_Rocksalt.yaml \
    --sampled_data_file_path ./output/Case3_Rocksalt/sampled_data/sample_sobol_size4_seed21.h5 \
    --get_field_component_index "[0, 1, 2]" \
    --sort_by_index 2 \
    --run_mode ensemble \
    --parallel True \
    --n_jobs 4 \
    --keep_vtu False \
    --path_to_save_results_hdf5_file default \
    --save_sampled_data True \