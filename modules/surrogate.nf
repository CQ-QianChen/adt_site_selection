include { toYaml; engineName } from './utils.nf'

process SURROGATE {
    conda "$projectDir/surrogate/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/surrogate_models" }, mode: 'copy'

    input:
    val config
    val uncertain_parameters
    val scenario_name
    val hash8
    val training_sampling_config
    val validation_sampling_config
    path geometry
    path rock_data
    path nuclide_water_diffusivity_data
    path training_data
    path validation_data

    output:
    path "${engineName(config, training_sampling_config, validation_sampling_config, hash8)}", emit: engine

    script:
    def engine_name = engineName(config, training_sampling_config, validation_sampling_config, hash8)
    """
    echo "${toYaml(config)}" > config.yaml
    echo "${toYaml([uncertain_parameters: uncertain_parameters])}" > uncertain_parameters.yaml

    python -m nuctrans_fastuq.surrogate.train_surrogate \
      --config config.yaml \
      --scenario_name ${scenario_name} \
      --uncertain_parameters uncertain_parameters.yaml \
      --training_data ${training_data} \
      --validation_data ${validation_data} \
      --scenario_folder . \
      --engine_folder ${engine_name}
    """
}
