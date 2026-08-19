#!/usr/bin/env nextflow

def toYaml(data) {
    def yb = new groovy.yaml.YamlBuilder()
    yb.call(data)
    return yb.toString()
}

def canonicalize(data) {
    if (data instanceof Map) {
        return new TreeMap(data.collectEntries { k, v -> [(k): canonicalize(v)] })
    } else if (data instanceof List) {
        return data.collect { canonicalize(it) }
    }
    return data
}

def computeHash8(data) {
    def json = groovy.json.JsonOutput.toJson(canonicalize(data))
    def digest = java.security.MessageDigest.getInstance("SHA-256").digest(json.getBytes("UTF-8"))
    def hex = digest.collect { String.format("%02x", it) }.join()
    return hex[-8..-1]
}

def resolveSamplingCombinations(cfg) {
    def combos = []
    cfg.sample_size.each { sz ->
        cfg.sampling_method.each { m ->
            cfg.seed.each { sd ->
                combos << [sample_size: sz, sampling_method: m, seed: sd, save_file_type: cfg.save_file_type]
            }
        }
    }
    return combos
}

process COMMUNICATION_SDH {
    conda "$moduleDir/communication_sdh/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val params_yaml
    val case_name
    val hash8

    output:
    path "rock_data", emit: rock_data
    path "site_data", emit: site_data
    path "geometry", emit: geometry
    path "params.yaml", emit: params

    script:
    """
    echo "${params_yaml}" > params.yaml

    python -m smart_data_hub.export_data \
      --config params.yaml \
      --path_to_save_rock_yaml rock_data \
      --path_to_save_site_yaml site_data \
      --path_to_save_site_geometry geometry
    """
}

process SAMPLING {
    conda "$moduleDir/sampling/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    path script
    val params_yaml
    val case_name
    val hash8
    path geometry
    path rock_data
    path nuclide_sorption_data
    path nuclide_water_diffusivity_data
    val sample_size
    val sampling_method
    val seed
    val save_file_type

    output:
    path "sampled_data/*", emit: sampled_data

    script:
    """
    echo "${params_yaml}" > params.yaml

    mkdir -p sampled_data

    python ${script} \
      --config params.yaml \
      --sample_size ${sample_size} \
      --sampling_method ${sampling_method} \
      --seed ${seed} \
      --path_to_geometry_data ${geometry} \
      --path_to_rock_data ${rock_data} \
      --path_to_sorption_data ${nuclide_sorption_data} \
      --path_to_diffusivity_data ${nuclide_water_diffusivity_data} \
      --path_to_save_sampled_data sampled_data \
      --save_file_type ${save_file_type}
    """
}

process COMMUNICATION_NTD {
    conda "$moduleDir/communication_ntd/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val params_yaml
    val case_name
    val hash8
    path site_data
    val site_name

    output:
    path "nuclide_sorption_data", emit: nuclide_sorption_data
    path "nuclide_species_data", emit: nuclide_species_data
    path "nuclide_water_diffusivity_data", emit: nuclide_water_diffusivity_data
    path "nuclide_emitted_energy_data", emit: nuclide_emitted_energy_data

    script:
    """
    echo "${params_yaml}" > params.yaml

    python -m nuctransportdb.export_data \
      --config params.yaml \
      --path_to_site_yaml_file ${site_data}/${site_name}.yaml \
      --path_to_save_sorption_data nuclide_sorption_data \
      --path_to_save_nuclide_species_data nuclide_species_data \
      --path_to_save_nuclide_water_diffusivity_data nuclide_water_diffusivity_data \
      --path_to_save_nuclide_emitted_energy_data nuclide_emitted_energy_data
    """
}

workflow {
    def case_name = file(params.config_file).getBaseName()
    def case_cfg = new groovy.yaml.YamlSlurper().parseText(file(params.config_file).text) as Map
    def hash8 = computeHash8(case_cfg.findAll { k, v -> k != 'sampling' })

    def sampling_combos = resolveSamplingCombinations(case_cfg.sampling)
    def resolved_case_cfg = case_cfg + [sampling: sampling_combos]

    COMMUNICATION_SDH(toYaml(resolved_case_cfg), case_name, hash8)

    COMMUNICATION_NTD(toYaml(case_cfg), case_name, hash8, COMMUNICATION_SDH.out.site_data, case_cfg.site_name)

    def sampling_cfg = [uncertain_parameters: case_cfg.uncertain_parameters]
    def sampling_combinations = Channel.fromList(sampling_combos).multiMap { combo ->
        size: combo.sample_size
        method: combo.sampling_method
        seed: combo.seed
    }

    SAMPLING(
        file("$moduleDir/sampling/sampling_func.py"),
        toYaml(sampling_cfg),
        case_name,
        hash8,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data,
        sampling_combinations.size,
        sampling_combinations.method,
        sampling_combinations.seed,
        case_cfg.sampling.save_file_type
    )
}
