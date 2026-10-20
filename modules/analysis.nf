include { toYaml } from './utils.nf'

process ANALYSIS {
    conda "$projectDir/analysis/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}" }, mode: 'copy'

    input:
    path analyze
    val analysis_config
    val setup
    val scenario_name
    val hash8
    path engine_dir
    path propagation
    path sensitivity
    path geometry
    path rock_data
    path nuclide_water_diffusivity_data

    output:
    path "analysis", emit: analysis

    script:
    """
    echo "${toYaml(analysis_config)}" > analysis.yaml
    echo "${toYaml(setup)}" > setup.yaml

    python ${analyze} \
      --propagation ${propagation} \
      --sensitivity ${sensitivity} \
      --geometry ${geometry}/rock_interface.yaml \
      --data ${geometry} ${rock_data} ${nuclide_water_diffusivity_data} \
      --engine ${engine_dir} \
      --setup setup.yaml \
      --hash ${hash8} \
      --config analysis.yaml \
      --output-dir analysis
    """
}
