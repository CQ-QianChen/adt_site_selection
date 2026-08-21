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

process MODEL {
    conda "$moduleDir/model/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val params_yaml
    val case_name
    val hash8
    path rock_data
    path site_data
    path geometry
    path nuclide_emitted_energy_data
    path nuclide_species_data
    path nuclide_sorption_data
    path nuclide_water_diffusivity_data
    path sampled_data
    val n_jobs
    val parallel
    val keep_vtu
    val get_field_component_index
    val sort_by_index

    output:
    path "model_results/*", type: 'any', emit: model_results

    script:
    """
    echo "${params_yaml}" > model_config.yaml

    mkdir -p model_results

    python -m yaml2nuctrans.get_model.ogs_model \
      --output_directory model_results/${sampled_data.baseName} \
      --rock_data_folder_path ${rock_data} \
      --site_folder_path ${site_data} \
      --geometry_folder_path ${geometry} \
      --emitted_energy_folder_path ${nuclide_emitted_energy_data} \
      --species_type_folder_path ${nuclide_species_data} \
      --sorption_data_folder_path ${nuclide_sorption_data} \
      --nuclide_water_diffusivity_folder_path ${nuclide_water_diffusivity_data} \
      --model_config_path model_config.yaml \
      --sampled_data_file_path ${sampled_data} \
      --get_field_component_index "${get_field_component_index}" \
      --sort_by_index ${sort_by_index} \
      --run_mode ensemble \
      --parallel ${parallel ? 'True' : 'False'} \
      --n_jobs ${n_jobs} \
      --keep_vtu ${keep_vtu ? 'True' : 'False'} \
      --path_to_save_results_hdf5_file model_results/ensemble_results_with_${sampled_data.baseName}.h5 \
      --save_sampled_data True

    rmdir model_results/${sampled_data.baseName} 2>/dev/null || true
    """
}

workflow {
    def case_name = file(params.config_file).getBaseName()
    def cfg = new groovy.yaml.YamlSlurper().parseText(file(params.config_file).text) as Map
    def case_cfg = cfg.findAll { k, v -> !(k in ['sampling', 'simulator_config']) }
    def hash8 = computeHash8(case_cfg)

    COMMUNICATION_SDH(toYaml(case_cfg), case_name, hash8)

    COMMUNICATION_NTD(toYaml(case_cfg), case_name, hash8, COMMUNICATION_SDH.out.site_data, cfg.site_name)

    def sampling_cfg = [uncertain_parameters: cfg.uncertain_parameters]
    SAMPLING(
        file("$moduleDir/sampling/sampling_func.py"),
        toYaml(sampling_cfg),
        case_name,
        hash8,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data,
        cfg.sampling.sample_size,
        cfg.sampling.sampling_method,
        cfg.sampling.seed,
        cfg.sampling.save_file_type
    )

    MODEL(
        toYaml(cfg.model_config),
        case_name,
        hash8,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_SDH.out.site_data,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_NTD.out.nuclide_emitted_energy_data,
        COMMUNICATION_NTD.out.nuclide_species_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data,
        SAMPLING.out.sampled_data,
        cfg.simulator_config.n_jobs,
        cfg.simulator_config.parallel,
        cfg.simulator_config.keep_vtu,
        cfg.model_config.get_field_component_index,
        cfg.simulator_config.sort_by_index
    )
}
