include { toYaml; caseConfig; caseHash } from './utils.nf'

process CREATE_DATASTORE {
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name

    output:
    path "params.yaml"
    val hash8, emit: hash8

    script:
    def case_cfg = caseConfig(config)
    hash8 = caseHash(config)
    """
    echo "${toYaml(case_cfg)}" > params.yaml
    """

}
