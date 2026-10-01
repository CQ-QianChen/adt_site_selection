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

process CREATE_DATASTORE {
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name

    output:
    path "params.yaml"
    val hash8, emit: hash8

    script:
    def case_cfg = config.findAll { k, v -> !(k in ['sampling_config', 'simulator_config']) }
    hash8 = computeHash8(case_cfg)
    """
    echo "${toYaml(case_cfg)}" > params.yaml
    """

}

process COMMUNICATION_SDH {
    conda "$moduleDir/communication_sdh/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
    val case_name
    val hash8

    output:
    path "rock_data", emit: rock_data
    path "site_data", emit: site_data
    path "geometry", emit: geometry

    script:
    """
    echo "${toYaml(config)}" > config.yaml

    python -m smart_data_hub.export_data \
      --config config.yaml \
      --path_to_save_rock_yaml rock_data \
      --path_to_save_site_yaml site_data \
      --path_to_save_site_geometry geometry
    """
}

process COMMUNICATION_NTD {
    conda "$moduleDir/communication_ntd/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
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
    echo "${toYaml(config)}" > config.yaml

    python -m nuctransportdb.export_data \
      --config config.yaml \
      --path_to_site_yaml_file ${site_data}/${site_name}.yaml \
      --path_to_save_sorption_data nuclide_sorption_data \
      --path_to_save_nuclide_species_data nuclide_species_data \
      --path_to_save_nuclide_water_diffusivity_data nuclide_water_diffusivity_data \
      --path_to_save_nuclide_emitted_energy_data nuclide_emitted_energy_data
    """
}

process SAMPLING {
    conda "$moduleDir/sampling/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    path script
    val uncertain_parameters
    val sampling_config
    val case_name
    val hash8
    path geometry
    path rock_data
    path nuclide_sorption_data
    path nuclide_water_diffusivity_data

    output:
    path "sampled_data/*", emit: sampled_data

    script:
    """
    echo "${toYaml([uncertain_parameters: uncertain_parameters])}" > config.yaml

    mkdir -p sampled_data

    python ${script} \
      --config config.yaml \
      --sample_size ${sampling_config.sample_size} \
      --sampling_method ${sampling_config.sampling_method} \
      --seed ${sampling_config.seed} \
      --path_to_geometry_data ${geometry} \
      --path_to_rock_data ${rock_data} \
      --path_to_sorption_data ${nuclide_sorption_data} \
      --path_to_diffusivity_data ${nuclide_water_diffusivity_data} \
      --path_to_save_sampled_data sampled_data \
      --save_file_type ${sampling_config.save_file_type}
    """
}

process MODEL {
    conda "$moduleDir/model/environment.yaml"
    publishDir { "$projectDir/datastore/${case_name}/${hash8}" }, mode: 'copy'

    input:
    val config
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
    val simulator_config

    output:
    path "model_results/*", type: 'any', emit: model_results

    script:
    """
    echo "${toYaml(config)}" > config.yaml

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
      --model_config_path config.yaml \
      --sampled_data_file_path ${sampled_data} \
      --get_field_component_index "${config.get_field_component_index}" \
      --sort_by_index ${simulator_config.sort_by_index} \
      --run_mode ensemble \
      --parallel ${simulator_config.parallel ? 'True' : 'False'} \
      --n_jobs ${simulator_config.n_jobs} \
      --keep_vtu ${simulator_config.keep_vtu ? 'True' : 'False'} \
      --path_to_save_results_hdf5_file model_results/ensemble_results_with_${sampled_data.baseName}.h5 \
      --save_sampled_data True

    rmdir model_results/${sampled_data.baseName} 2>/dev/null || true
    """
}

def defaultsConfig() {
    return [
        communication_sdh: [
            sampling_functions_by_property: [
                electrical_resistivity: 'generate_lognorm',
                intrinsic_permeability: 'generate_lognorm'
            ]
        ],
        sampling_config: [
            save_file_type: 'HDF5'
        ]
    ]
}

workflow {
    def case_name = file(params.config_file).getBaseName()
    def case_config = new groovy.yaml.YamlSlurper().parseText(file(params.config_file).text) as Map

    case_config.communication_sdh = defaultsConfig().communication_sdh + case_config.communication_sdh

    CREATE_DATASTORE(
        case_config,
        case_name
    )

    COMMUNICATION_SDH(
        case_config.communication_sdh,
        case_name,
        CREATE_DATASTORE.out.hash8
    )

    COMMUNICATION_NTD(
        case_config.communication_ntd,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.site_data,
        case_config.communication_sdh.site_name
    )

    def training_samples = SAMPLING(
        file("$moduleDir/sampling/sampling_func.py"),
        case_config.uncertain_parameters,
        defaultsConfig().sampling_config + case_config.sampling_config.training,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data
    )

    def validation_samples = SAMPLING(
        file("$moduleDir/sampling/sampling_func.py"),
        case_config.uncertain_parameters,
        defaultsConfig().sampling_config + case_config.sampling_config.validation,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data
    )

    MODEL(
        case_config.model,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_SDH.out.site_data,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_NTD.out.nuclide_emitted_energy_data,
        COMMUNICATION_NTD.out.nuclide_species_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data,
        training_samples.sampled_data,
        case_config.simulator_config
    )

    MODEL(
        case_config.model,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_SDH.out.site_data,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_NTD.out.nuclide_emitted_energy_data,
        COMMUNICATION_NTD.out.nuclide_species_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data,
        validation_samples.sampled_data,
        case_config.simulator_config
    )
}
