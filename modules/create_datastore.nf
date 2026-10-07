include { toYaml; computeHash8 } from './utils.nf'

process CREATE_DATASTORE {
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name

    output:
    path "params.yaml"
    val hash8, emit: hash8

    script:
    def case_cfg = config.findAll { k, v -> !(k in ['sampling_config', 'simulator_config']) }
    hash8 = computeHash8(case_cfg)
    """
    echo "${toYaml(case_cfg)}" > params.yaml
    """

}
