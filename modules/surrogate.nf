include { toYaml; computeHash8 } from './utils.nf'

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
    // surrogate id: its own settings plus the training and validation samples it sees
    def surrogate_hash8 = computeHash8([
        surrogate: config,
        training: training_sampling_config,
        validation: validation_sampling_config
    ])
    def engine_name = "engine_${config.type.toLowerCase()}_${surrogate_hash8}"
    """
    echo "${toYaml(config)}" > config.yaml
    echo "${toYaml([uncertain_parameters: uncertain_parameters])}" > uncertain_parameters.yaml

    python $projectDir/surrogate/train_surrogate.py \
      --config config.yaml \
      --case_name ${case_name} \
      --uncertain_parameters uncertain_parameters.yaml \
      --training_data ${training_data} \
      --validation_data ${validation_data} \
      --case_folder . \
      --engine_folder surrogate_models/${engine_name}
    """
}
