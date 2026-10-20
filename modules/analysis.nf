include { toYaml } from './utils.nf'

process ANALYSIS {
    conda "$projectDir/analysis/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${run_hash}" }, mode: 'copy'

    input:
    path analyze
    val scenario_config
    val scenario_name
    val hash8
    val run_hash
    path engine_dir
    path propagation
    path sensitivity
    path geometry
    path rock_data
    path nuclide_water_diffusivity_data

    output:
    path "analysis", emit: analysis
    path "scenario.yaml", emit: scenario

    script:
    """
    echo "${toYaml(scenario_config)}" > scenario.yaml

    python ${analyze} \
      --propagation ${propagation} \
      --sensitivity ${sensitivity} \
      --geometry ${geometry}/rock_interface.yaml \
      --data ${geometry} ${rock_data} ${nuclide_water_diffusivity_data} \
      --engine ${engine_dir} \
      --scenario scenario.yaml \
      --hash ${hash8} \
      --run-hash ${run_hash} \
      --output-dir analysis
    """
}
