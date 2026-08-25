# Active Digital Twin for Nuclear Waste Disposal Site Selection

A workflow for uncertainty-quantified I-129 nuclide transport simulations
across candidate host-rock formations, supporting nuclear waste disposal
site selection. See [Usage](#usage) for setup and run instructions.

This computational study was developed using the
[SHOWME.how](https://mbd-rwth.github.io/showmehow/) approach.

<p align="left">
  <img src="assets/showmehow_logo.svg" alt="SHOWME.how" height="64">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/logo_mbd_rgb.png" alt="MBD – RWTH Aachen University" height="64">
</p>

## Case Overview
| Case                                                                                                          | Host rock   | Region            | Uncertain parameters                                                                                             | Idea: Plot for paper                                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------|-------------|-------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [Case1_Claystone_Lower-Saxony](./scenario/Case1_Claystone_Lower-Saxony.yaml) | Claystone   | Lower Saxony      | `porosity`, `density`, `nuclide_water_diffusivity` for the host rock unit and its overlying and underlying units | Concentration time evolution at the upper and lower host rock interfaces Or the concentration spatial distribution at a certain time |
| [Case2_Claystone_uq_geometry](./scenario/Case2_Claystone_uq_geometry.yaml)   | Claystone   | Lower Saxony      | Depths of the upper and lower host rock interfaces                                                               | Nuclide flux time evolution at the upper and lower interfaces                                                                        |
| [Case3_Claystone](./scenario/Case3_Claystone.yaml)                           | Claystone   | Lower Saxony      | `porosity`, `density`, `nuclide_water_diffusivity` for the host rock unit and its overlying and underlying units | Nuclide flux time evolution at the upper and lower interfaces                                                                        |
| [Case3_Crystalline](./scenario/Case3_Crystalline.yaml)                       | Crystalline | Baden-Württemberg | `porosity`, `density`, `nuclide_water_diffusivity` for the host rock unit and its overlying and underlying units | Nuclide flux time evolution at the upper and lower interfaces                                                                        |
| [Case3_Rocksalt](./scenario/Case3_Rocksalt.yaml)                             | Rock salt   | Saxony-Anhalt     | `porosity`, `density`, `nuclide_water_diffusivity` for the host rock unit and its overlying and underlying units | Nuclide flux time evolution at the upper and lower interfaces                                                                        |

- Note: Flux values span 1e-14–1e-20 (mol/L)·(m/s); treat 1e-20 as the minimum
valid value during post-processing.

## Usage

### Setup

The workflow is orchestrated with [Nextflow](https://www.nextflow.io/). Create
the environment that provides `nextflow` itself from the root
[`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
conda activate adt_site_selection
```

Nextflow builds and caches each process's own conda environment automatically
on first run, so no further setup is needed.

### Running the workflow

The pipeline is defined in [`main.nf`](./main.nf) and takes a single scenario
config file as input:

```bash
nextflow run main.nf --config_file scenario/Case3_Claystone.yaml
```

Any file in [`scenario/`](./scenario/) can be passed this way, e.g.
`scenario/Case1_Claystone_Lower-Saxony.yaml`, `scenario/Case3_Rocksalt.yaml`,
etc. A scenario file has six top-level sections:

| Key                 | Feeds                          | Contains                                                          |
|----------------------|--------------------------------|--------------------------------------------------------------------|
| `communication_sdh`  | `COMMUNICATION_SDH`            | Site name, tag dict, rock-unit merging/depth setup                |
| `communication_ntd`  | `COMMUNICATION_NTD`            | Nuclides to consider                                               |
| `uncertain_parameters`| `SAMPLING`                    | Which rock/nuclide/geometry properties to sample                  |
| `sampling_config`    | `SAMPLING`                     | `sample_size`, `sampling_method`, `seed`, `save_file_type`         |
| `model`               | `MODEL`                        | OGS model/simulation setup                                        |
| `simulator_config`   | `MODEL`                        | `n_jobs`, `parallel`, `keep_vtu`, `sort_by_index`                  |

Running the pipeline executes, in order: `CREATE_DATASTORE` (records the
scenario's config), `COMMUNICATION_SDH` and `COMMUNICATION_NTD` (fetch
site/rock/nuclide data), `SAMPLING` (draws parameter samples), and `MODEL`
(runs the OGS ensemble).

### Datastore

Every run publishes into `datastore/<case_name>/<hash8>/`:

```
datastore/<case_name>/<hash8>/
├── params.yaml                        # the config used to compute hash8
├── rock_data/                         # from COMMUNICATION_SDH
├── site_data/                         # from COMMUNICATION_SDH
├── geometry/                          # from COMMUNICATION_SDH
├── nuclide_sorption_data/             # from COMMUNICATION_NTD
├── nuclide_species_data/              # from COMMUNICATION_NTD
├── nuclide_water_diffusivity_data/    # from COMMUNICATION_NTD
├── nuclide_emitted_energy_data/       # from COMMUNICATION_NTD
├── sampled_data/                      # one file per sampling_config used, from SAMPLING
└── model_results/                     # one file per sampled_data file, from MODEL
```

`hash8` is a hash of `communication_sdh`, `communication_ntd`,
`uncertain_parameters`, and `model`; `sampling_config` and
`simulator_config` are excluded. So changing the site, rock/nuclide
selection, or model setup creates a new `hash8` (a new scenario directory),
while changing only `sampling_config` or `simulator_config` reuses the same
directory. `SAMPLING`/`MODEL` name their output files after the sampling
parameters (e.g. `sample_sobol_size4_seed21.h5`), so this just adds a new
sample/result file rather than overwriting the scenario.
