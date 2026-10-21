process REPORT {
    conda "$projectDir/report/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${uq_hash}/analysis/${analysis_hash}" }, mode: 'copy'

    input:
    path build
    path fonts
    path analysis
    val scenario_name
    val hash8
    val uq_hash
    val analysis_hash

    output:
    path "report.pdf", emit: report

    script:
    """
    python ${build} --analysis ${analysis} --scenario ${analysis}/scenario.yaml --output report.pdf
    """
}
