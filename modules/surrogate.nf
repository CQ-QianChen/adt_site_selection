include { toYaml; engineName } from './utils.nf'

process SURROGATE {
    conda "$projectDir/surrogate/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val uncertain_parameters
    val case_name
    val hash8
    val training_sampling_config
    val validation_sampling_config
    path geometry
    path rock_data
    path nuclide_water_diffusivity_data
    path training_data
    path validation_data

    output:
    path "surrogate_models", emit: surrogate_models

    script:
    def engine_name = engineName(config, training_sampling_config, validation_sampling_config)
    """
    echo "${toYaml(config)}" > config.yaml
    echo "${toYaml([uncertain_parameters: uncertain_parameters])}" > uncertain_parameters.yaml

    python -m nuctrans_fastuq.surrogate.train_surrogate \
      --config config.yaml \
      --case_name ${case_name} \
      --uncertain_parameters uncertain_parameters.yaml \
      --training_data ${training_data} \
      --validation_data ${validation_data} \
      --case_folder . \
      --engine_folder surrogate_models/${engine_name}
    """
}
