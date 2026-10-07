include { toYaml } from './utils.nf'

process COMMUNICATION_SDH {
    conda "$projectDir/communication_sdh/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name
    val hash8

    output:
    path "rock_data", emit: rock_data
    path "site_data", emit: site_data
    path "geometry", emit: geometry

    script:
    """
    echo "${toYaml(config)}" > config.yaml

    python -m smart_data_hub.export_data \
      --config config.yaml \
      --path_to_save_rock_yaml rock_data \
      --path_to_save_site_yaml site_data \
      --path_to_save_site_geometry geometry
    """
}
