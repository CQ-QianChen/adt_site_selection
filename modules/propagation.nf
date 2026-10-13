include { toYaml } from './utils.nf'

process PROPAGATION {
    conda "$projectDir/surrogate/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}/decision_support_uq" }, mode: 'copy'

    input:
    val propagation_opts
    val case_name
    val hash8
    val engine_name
    path engine_dir

    output:
    path "propagation_${engine_name}_*.h5", emit: propagation

    script:
    def method = propagation_opts.sampling_method ? "_${propagation_opts.sampling_method}" : ""
    def out = "propagation_${engine_name}_n${propagation_opts.n_samples}_seed${propagation_opts.seed}${method}.h5"
    """
    echo "${toYaml(propagation_opts)}" > propagation_opts.yaml

    python -m nuctrans_fastuq.uq.propagation \
      --engine ${engine_dir}/engine.h5 \
      --config propagation_opts.yaml \
      --output ${out}
    """
}
