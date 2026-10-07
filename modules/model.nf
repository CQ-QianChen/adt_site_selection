include { toYaml } from './utils.nf'

process MODEL {
    conda "$projectDir/model/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name
    val hash8
    path rock_data
    path site_data
    path geometry
    path nuclide_emitted_energy_data
    path nuclide_species_data
    path nuclide_sorption_data
    path nuclide_water_diffusivity_data
    path sampled_data
    val simulator_config

    output:
    path "model_results/*", type: 'any', emit: model_results

    script:
    """
    echo "${toYaml(config)}" > config.yaml

    mkdir -p model_results

    python -m yaml2nuctrans.get_model.ogs_model \
      --output_directory model_results/${sampled_data.baseName} \
      --rock_data_folder_path ${rock_data} \
      --site_folder_path ${site_data} \
      --geometry_folder_path ${geometry} \
      --emitted_energy_folder_path ${nuclide_emitted_energy_data} \
      --species_type_folder_path ${nuclide_species_data} \
      --sorption_data_folder_path ${nuclide_sorption_data} \
      --nuclide_water_diffusivity_folder_path ${nuclide_water_diffusivity_data} \
      --model_config_path config.yaml \
      --sampled_data_file_path ${sampled_data} \
      --get_field_component_index "${config.get_field_component_index}" \
      --sort_by_index ${simulator_config.sort_by_index} \
      --run_mode ${simulator_config.run_mode} \
      --parallel ${simulator_config.parallel ? 'True' : 'False'} \
      --n_jobs ${simulator_config.n_jobs} \
      --keep_vtu ${simulator_config.keep_vtu ? 'True' : 'False'} \
      --path_to_save_results_hdf5_file model_results/ensemble_results_with_${sampled_data.baseName}.h5 \
      --save_sampled_data ${simulator_config.save_sampled_data}

    rmdir model_results/${sampled_data.baseName} 2>/dev/null || true
    """
}
