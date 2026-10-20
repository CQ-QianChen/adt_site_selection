include { toYaml } from './utils.nf'

process PROPAGATION {
    conda "$projectDir/surrogate/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${run_hash}" }, mode: 'copy'

    input:
    val propagation_opts
    val scenario_name
    val hash8
    val run_hash
    path engine_dir

    output:
    path "propagation.h5", emit: propagation

    script:
    """
    echo "${toYaml(propagation_opts)}" > propagation_opts.yaml

    python -m nuctrans_fastuq.uq.propagation \
      --engine ${engine_dir}/engine.h5 \
      --config propagation_opts.yaml \
      --output propagation.h5
    """
}
