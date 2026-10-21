process REPORT {
    conda "$projectDir/decision_support_report/environment.yaml"
    publishDir { "$projectDir/datastore/${scenario_name}/${hash8}/decision_support/${uq_hash}/analysis/${analysis_hash}" }, mode: 'copy'

    input:
    path build
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

// the comparison of several runs: the datastore folder of the analysis of each site, in the order of the labels (not staged: the hashes in its path name the content)
process COMPARE {
    conda "$projectDir/decision_support_report/environment.yaml"
    publishDir { "$projectDir/datastore/comparisons/${name}" }, mode: 'copy'

    input:
    path compare
    path build
    path regions
    val name
    val title
    val labels
    val run_dirs

    output:
    path "comparison.pdf", emit: report
    path "figures", emit: figures

    script:
    def runs = [labels, run_dirs].transpose().collect { label, dir -> "--run \"${label}=${dir}\"" }.join(' ')
    """
    PYTHONPATH=. python ${compare} ${runs} --regions ${regions} --title "${title}" --output-dir .
    """
}
