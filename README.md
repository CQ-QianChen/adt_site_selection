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

## Scenarios

| Scenario | Host rock | Region | Uncertain parameters |
|------|-----------|--------|----------------------|
| [Case1_Claystone_Lower-Saxony](./scenario/Case1_Claystone_Lower-Saxony.yaml) | Claystone | Lower Saxony | `porosity`, `density`, `nuclide_water_diffusivity` of the host rock and its neighbouring units |
| [Case2_Claystone_uq_geometry](./scenario/Case2_Claystone_uq_geometry.yaml) | Claystone | Lower Saxony | depths of the host rock interfaces |
| [Case3_Claystone](./scenario/Case3_Claystone.yaml) | Claystone | Lower Saxony | `porosity`, `density`, `nuclide_water_diffusivity` of the host rock and its neighbouring units |
| [Case3_Crystalline](./scenario/Case3_Crystalline.yaml) | Crystalline | Baden-Württemberg | `porosity`, `density`, `nuclide_water_diffusivity` of the host rock and its neighbouring units |
| [Case3_Rocksalt](./scenario/Case3_Rocksalt.yaml) | Rock salt | Saxony-Anhalt | `porosity`, `density`, `nuclide_water_diffusivity` of the host rock and its neighbouring units |

## Usage

### Setup

The workflow is orchestrated with [Nextflow](https://www.nextflow.io/). Create
the environment that provides `nextflow` from [`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
conda activate adt_site_selection
```

Nextflow builds and caches the conda environment of each process on first run.

### Running a scenario

```bash
nextflow run main.nf --config_file scenario/Case3_Claystone.yaml
```

Any file in [`scenario/`](./scenario/) can be passed. The workflow in [`main.nf`](./main.nf) runs the following steps in order:

| Step | Does | Runs |
|------|------|------|
| `CREATE_DATASTORE` | records the configuration of the scenario | always |
| `COMMUNICATION_SDH`, `COMMUNICATION_NTD` | fetch the site, rock and nuclide data | always |
| `SAMPLING`, `MODEL`, `SURROGATE` | draw the training and validation datasets, run the OGS ensemble, train the surrogate | if there is no surrogate engine yet |
| `SENSITIVITY`, `PROPAGATION` | compute the Sobol indices, propagate the input uncertainty | if the scenario asks for them |
| `ANALYSIS`, `REPORT` | compute the statistics, draw the figures, write the PDF report | if the scenario describes an analysis |

### Scenario file

| Key | Description |
|-----|-------------|
| `name` | the scenario name, also its datastore folder |
| `communication_sdh` | the site, and how its rock units and depths are set up |
| `communication_ntd` | the nuclides to consider |
| `uncertain_parameters` | the properties that are sampled |
| `model` | the setup of the OGS model |
| `simulator_config` | how the OGS ensemble is run |
| `surrogate` | the surrogate type, its training and validation datasets, and the output it predicts |
| `uq` | which uncertainty analyses to run (sensitivity, propagation) |
| `analysis` | what the report shows, for example the threshold and the locations. It can be changed without recomputing the surrogate |

### Datastore

Results are written to `datastore/`:

```
datastore/<scenario_name>/<hash8>/
├── params.yaml                        # the configuration that hash8 is computed from
├── rock_data/, site_data/, geometry/  # from COMMUNICATION_SDH
├── nuclide_*_data/                    # from COMMUNICATION_NTD
├── sampled_data/, model_results/      # training and validation datasets
├── surrogate_models/<engine>/         # the surrogate engine and its validation metrics
└── decision_support/<uq_hash>/
    ├── sensitivity.h5, propagation.h5
    └── analysis/<analysis_hash>/      # stats.json, figures/, report.pdf
```

Each level is named by a hash of its configuration, so a result is reused when the
config behind it is unchanged:

| Folder | Hash of |
|--------|---------|
| `<hash8>` | the scenario without `name`, `simulator_config`, `surrogate`, `uq`, `analysis` |
| `<engine>` | the scenario, the `surrogate` settings and their sampling |
| `<uq_hash>` | the engine and the `uq` options |
| `<analysis_hash>` | `analysis` without `figures`, `text`, `style`, `plot_options` |

For example, changing the `threshold` in the [`analysis`](#scenario-file) block of the scenario
file gives a new analysis folder; the surrogate engine and the UQ results are reused.

### Running in containers

```bash
nextflow run main.nf --config_file scenario/Case3_Claystone.yaml --containerized
```

Each step runs in the container image of its module instead of a conda environment. The images fix every package version,
so the software is the same on every machine. `-profile apptainer` uses Apptainer instead of Docker.

### Comparing scenarios

```bash
nextflow run main.nf --comparison_file scenario/comparison/Case3.yaml
```

The comparison file lists the scenario files of the sites. All sites need the same threshold
(`analysis.qoi.threshold`), or none. Otherwise the run stops and says what to change.
The report is written to `datastore/comparisons/<name>/`.
