process REPORT {
    conda "$projectDir/report/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${run_hash}" }, mode: 'copy'

    input:
    path build
    path fonts
    path analysis
    path scenario
    val scenario_name
    val hash8
    val run_hash

    output:
    path "report.pdf", emit: report

    script:
    """
    python ${build} --analysis ${analysis} --scenario ${scenario} --output report.pdf
    """
}
