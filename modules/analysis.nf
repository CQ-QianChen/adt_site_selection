include { toYaml } from './utils.nf'

process ANALYSIS {
    conda "$projectDir/decision_support_analysis/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${uq_hash}/analysis" }, mode: 'copy'

    input:
    path analyze
    val scenario_config
    val scenario_name
    val hash8
    val uq_hash
    val analysis_hash
    path engine_dir
    path propagation
    path sensitivity
    path geometry
    path rock_data
    path nuclide_water_diffusivity_data

    output:
    path "${analysis_hash}", emit: analysis

    script:
    """
    mkdir ${analysis_hash}
    echo "${toYaml(scenario_config)}" > ${analysis_hash}/scenario.yaml

    python ${analyze} \
      --propagation ${propagation} \
      --sensitivity ${sensitivity} \
      --geometry ${geometry}/rock_interface.yaml \
      --data ${geometry} ${rock_data} ${nuclide_water_diffusivity_data} \
      --engine ${engine_dir} \
      --scenario ${analysis_hash}/scenario.yaml \
      --hash ${hash8} \
      --uq-hash ${uq_hash} \
      --analysis-hash ${analysis_hash} \
      --output-dir ${analysis_hash}
    """
}
