include { toYaml } from './utils.nf'

process SENSITIVITY {
    conda "$projectDir/surrogate/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}/decision_support_uq" }, mode: 'copy'

    input:
    val sensitivity_opts
    val case_name
    val hash8
    val engine_name
    path engine_dir

    output:
    path "sensitivity_${engine_name}.h5", emit: sensitivity

    script:
    """
    echo "${toYaml(sensitivity_opts)}" > sensitivity_opts.yaml

    python -m nuctrans_fastuq.uq.sensitivity \
      --engine ${engine_dir}/engine.h5 \
      --config sensitivity_opts.yaml \
      --output sensitivity_${engine_name}.h5
    """
}
