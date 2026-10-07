include { toYaml } from './utils.nf'

process COMMUNICATION_NTD {
    conda "$projectDir/communication_ntd/environment.yaml"
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
