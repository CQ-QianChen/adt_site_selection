include { toYaml } from './utils.nf'

process SENSITIVITY {
    conda "$projectDir/decision_support_uq/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${uq_hash}" }, mode: 'copy'

    input:
    val sensitivity_opts
    val scenario_name
    val hash8
    val uq_hash
    path engine_dir

    output:
    path "sensitivity.h5", emit: sensitivity

    script:
    """
    echo "${toYaml(sensitivity_opts)}" > sensitivity_opts.yaml

    python -m nuctrans_fastuq.uq.sensitivity \
      --engine ${engine_dir}/engine.h5 \
      --config sensitivity_opts.yaml \
      --output sensitivity.h5
    """
}
