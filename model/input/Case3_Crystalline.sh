
python -m yaml2nuctrans.get_model.ogs_model \
    --output_directory ./output/Case3_Crystalline/model_results/ \
    --rock_data_folder_path ./output/Case3_Crystalline/rock_data \
    --site_folder_path ./output/Case3_Crystalline/site_data \
    --geometry_folder_path ./output/Case3_Crystalline/geometry \
    --emitted_energy_folder_path ./output/Case3_Crystalline/nuclide_emitted_energy_data \
    --species_type_folder_path ./output/Case3_Crystalline/nuclide_species_data \
    --sorption_data_folder_path ./output/Case3_Crystalline/nuclide_sorption_data \
    --nuclide_water_diffusivity_folder_path ./output/Case3_Crystalline/nuclide_water_diffusivity_data \
    --model_config_path ./model/input/config_Case3_Crystalline.yaml \
    --sampled_data_file_path ./output/Case3_Crystalline/sampled_data/sample_sobol_size4_seed21.h5 \
    --get_field_component_index "[0, 1, 2]" \
    --sort_by_index 2 \
    --run_mode ensemble \
    --parallel True \
    --n_jobs 4 \
    --keep_vtu False \
    --path_to_save_results_hdf5_file default \
    --save_sampled_data True \