include { toYaml } from './utils.nf'

process SURROGATE {
    conda "$projectDir/surrogate/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name
    val hash8
    path training_data
    path validation_data

    output:
    path "surrogate_models", emit: surrogate_models

    script:
    """
    echo "${toYaml(config)}" > config.yaml

    mkdir -p surrogate_models

    # placeholder: training and validation will be called here
    """
}
