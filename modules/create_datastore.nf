include { toYaml; scenarioConfig; scenarioHash } from './utils.nf'

process CREATE_DATASTORE {
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val scenario_name

    output:
    path "params.yaml"
    val hash8, emit: hash8

    script:
    def scenario_cfg = scenarioConfig(config)
    hash8 = scenarioHash(config)
    """
    echo "${toYaml(scenario_cfg)}" > params.yaml
    """

}
