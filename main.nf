#!/usr/bin/env nextflow

include { CREATE_DATASTORE } from './modules/create_datastore.nf'
include { COMMUNICATION_SDH } from './modules/communication_sdh.nf'
include { COMMUNICATION_NTD } from './modules/communication_ntd.nf'
include { SAMPLING as SAMPLING_TRAINING; SAMPLING as SAMPLING_VALIDATION } from './modules/sampling.nf'
include { MODEL as MODEL_TRAINING; MODEL as MODEL_VALIDATION } from './modules/model.nf'
include { SURROGATE } from './modules/surrogate.nf'

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
        ],
        model: [
            create_meshes: true,
            mesh_dimension: 1,
            quantity_of_interests: ['concentration', 'Flux'],
            process_variables: ['pressure', 'concentration'],
            secondary_variables: ['Flux'],
            sim_setup: [
                timeloop: [
                    processes: [
                        nonlinear_solver_name: 'basic_picard',
                        convergence_type: 'PerComponentDeltaX',
                        norm_type: 'NORM2',
                        time_discretization: 'BackwardEuler',
                        time_stepping: [
                            type: 'FixedTimeStepping',
                            t_initial: 0.0,
                            t_end: 31557600000000.0,
                            time_step: [31557600.0, 315576000.0, 3155760000.0, 31557600000.0, 315576000000.0],
                            repeat: [100, 90, 90, 90, 90]
                        ]
                    ],
                    output: [
                        repeat: ['10', '9', '9', '9', '9'],
                        each_steps: ['10', '10', '10', '10', '10']
                    ]
                ],
                non_linear_solver: [
                    name: 'basic_picard',
                    type: 'Picard',
                    max_iter: '10',
                    linear_solver: 'general_linear_solver'
                ],
                linear_solver: [
                    name: ['general_linear_solver', 'general_linear_solver', 'general_linear_solver'],
                    kind: ['lis', 'eigen', 'petsc'],
                    prefix: [null, null, 'hc'],
                    solver_type: ['cg', 'SparseLU', 'bcgs'],
                    precon_type: ['jacobi', 'ILUT', 'bjacobi'],
                    max_iteration_step: ['20000', '10000', '20000'],
                    error_tolerance: [1e-16, 1e-14, 1e-8]
                ]
            ],
            get_field_component_index: [0, 1, 2]
        ],
        simulator_config: [
            sort_by_index: 2,
            run_mode: 'ensemble',
            save_sampled_data: 'True'
        ],
        // type, output_qoi, z_interest, t_interest, qoi_floor, transform, pce_opts.q_norm and pce_opts.degree are per-case and have no default
        surrogate: [
            n_init: null,
            parallel: true,
            pce_opts: [
                type: 'aPCE',
                reg_method: 'FastARDExtended',
                dim_red_method: 'no',
                bootstrap_method: 'fast',
                n_bootstrap_itrs: 1,
                input_transform: 'user'
            ],
            gp_opts: [
                kernel_type: 'Matern',
                isotropy: false,
                auto_select: false,
                n_restarts: 10,
                noisy: true,
                nugget: 1.0e-9,
                normalize_x_method: 'norm',
                transform_y_method: 'norm',
                gpe_reg_method: 'LBFGS',
                dim_red_method: 'no',
                input_transform: 'user'
            ]
        ]
    ]
}

workflow {
    def case_name = file(params.config_file).getBaseName()
    def case_config = new groovy.yaml.YamlSlurper().parseText(file(params.config_file).text) as Map

    case_config.communication_sdh = defaultsConfig().communication_sdh + case_config.communication_sdh
    case_config.sampling_config.training = defaultsConfig().sampling_config + case_config.sampling_config.training
    case_config.sampling_config.validation = defaultsConfig().sampling_config + case_config.sampling_config.validation
    case_config.model = defaultsConfig().model + case_config.model + [project_name: case_name]
    case_config.simulator_config = defaultsConfig().simulator_config + case_config.simulator_config
    def surrogate_defaults = defaultsConfig().surrogate
    case_config.surrogate = surrogate_defaults + case_config.surrogate + [
        pce_opts: surrogate_defaults.pce_opts + (case_config.surrogate.pce_opts ?: [:]),
        gp_opts: surrogate_defaults.gp_opts + (case_config.surrogate.gp_opts ?: [:])
    ]
    
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

    SAMPLING_TRAINING(
        file("$moduleDir/sampling/sampling_func.py"),
        case_config.uncertain_parameters,
        case_config.sampling_config.training,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data
    )

    SAMPLING_VALIDATION(
        file("$moduleDir/sampling/sampling_func.py"),
        case_config.uncertain_parameters,
        case_config.sampling_config.validation,
        case_name,
        CREATE_DATASTORE.out.hash8,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_sorption_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data
    )

    MODEL_TRAINING(
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
        SAMPLING_TRAINING.out.sampled_data,
        case_config.simulator_config
    )

    MODEL_VALIDATION(
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
        SAMPLING_VALIDATION.out.sampled_data,
        case_config.simulator_config
    )

    SURROGATE(
        case_config.surrogate,
        case_config.uncertain_parameters,
        case_name,
        CREATE_DATASTORE.out.hash8,
        case_config.sampling_config.training,
        case_config.sampling_config.validation,
        COMMUNICATION_SDH.out.geometry,
        COMMUNICATION_SDH.out.rock_data,
        COMMUNICATION_NTD.out.nuclide_water_diffusivity_data,
        MODEL_TRAINING.out.model_results,
        MODEL_VALIDATION.out.model_results
    )
}
