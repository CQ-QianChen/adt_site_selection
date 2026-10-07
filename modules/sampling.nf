include { toYaml } from './utils.nf'

process SAMPLING {
    conda "$projectDir/sampling/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    path script
    val uncertain_parameters
    val sampling_config
    val case_name
    val hash8
    path geometry
    path rock_data
    path nuclide_sorption_data
    path nuclide_water_diffusivity_data

    output:
    path "sampled_data/*", emit: sampled_data

    script:
    """
    echo "${toYaml([uncertain_parameters: uncertain_parameters])}" > config.yaml

    mkdir -p sampled_data

    python ${script} \
      --config config.yaml \
      --sample_size ${sampling_config.sample_size} \
      --sampling_method ${sampling_config.sampling_method} \
      --seed ${sampling_config.seed} \
      --path_to_geometry_data ${geometry} \
      --path_to_rock_data ${rock_data} \
      --path_to_sorption_data ${nuclide_sorption_data} \
      --path_to_diffusivity_data ${nuclide_water_diffusivity_data} \
      --path_to_save_sampled_data sampled_data \
      --save_file_type ${sampling_config.save_file_type}
    """
}
