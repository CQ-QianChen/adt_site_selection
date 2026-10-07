# surrogate

Trains a surrogate (metamodel) of the nuclide-transport model on a set of
training simulations, validates it on a second set of simulations, and saves
the surrogate together with its validation metrics.

Supported surrogate types: polynomial chaos expansion (`pce`), Gaussian
process (`gpe`, scikit-learn; `gpe_torch`, PyTorch) and a combination of both
(`pce_gpe`). The training code is BayesValidRox, copied into `src/bayesvalidrox/`.

## Setup

```console
micromamba env create -f surrogate/environment.yaml
micromamba activate surrogate
```

## Inputs

| Input | What it is |
|---|---|
| Training data | HDF5 file with the model results of the training samples (`ensemble_results_with_sample_*.h5`, written by the model box). |
| Validation data | Same format, from independent samples. Optional: without it the surrogate is trained but not validated. |
| Case folder | Folder that contains `rock_data/`, `geometry/` and `nuclide_water_diffusivity_data/` (written by the communication boxes). Used to build the input distributions and to resolve layer names such as `Host_rock_top` into depths. |
| Uncertain parameters | yaml file with an `uncertain_parameters` block listing which parameters were sampled. |
| Settings | yaml file with the surrogate settings (see below). |

The uncertain-parameters file has the same structure that was used for
sampling:

```yaml
uncertain_parameters:
  rock_data:
    Host_rock:
      - porosity
      - density
  nuclide_water_diffusivity_data:
    Host_rock:
      - I-129
```

## Usage

```console
python surrogate/train_surrogate.py \
  --config settings.yaml \
  --case_name Case3_Claystone \
  --uncertain_parameters uncertain_parameters.yaml \
  --training_data model_results/ensemble_results_with_sample_sobol_size1000_seed21.h5 \
  --validation_data model_results/ensemble_results_with_sample_random_size500_seed21.h5 \
  --case_folder datastore/Case3_Claystone/<hash8> \
  --engine_folder surrogate_models/engine_pce_example
```

| Argument | |
|---|---|
| `--config` | Settings yaml. |
| `--case_name` | Name of the case, stored in the output. |
| `--uncertain_parameters` | Uncertain-parameters yaml. |
| `--training_data` | Training HDF5 file. |
| `--validation_data` | Validation HDF5 file (optional). |
| `--case_folder` | Folder with `rock_data/`, `geometry/` and `nuclide_water_diffusivity_data/`. |
| `--engine_folder` | Folder the results are written to. Created if missing. |

## Settings file

Example with every key (the values shown are examples, not recommendations):

```yaml
type: pce                        # pce | gpe | gpe_torch | pce_gpe
output_qoi: I-129Flux            # quantity of interest to model
z_interest:                      # depths to model, see below
  - Host_rock_top
  - Host_rock_bottom
t_interest: null                 # list of times, e.g. [1.0e+05, 5.0e+05]; null = all time steps
n_init: null                     # use only the first n training samples; null = all
qoi_floor: 1.0e-20               # values with a magnitude below this are set to zero
transform: null                  # output transform: null | div_by_max | standardize | minmax
parallel: true                   # train the output nodes in parallel

pce_opts:                        # used by pce and pce_gpe
  type: aPCE                     # aPCE | PCE
  reg_method: FastARDExtended    # FastARDExtended | FastARD | OLS | BRR | ARD | LARS | BCS | VBL | EBL
  q_norm: 0.6                    # hyperbolic truncation norm
  degree: [3, 4, 5]              # degrees to search; the best one is kept
  dim_red_method: 'no'           # 'no' | pca
  bootstrap_method: fast         # fast | normal
  n_bootstrap_itrs: 1
  input_transform: user

gp_opts:                         # used by gpe, gpe_torch and pce_gpe
  kernel_type: Matern            # Matern | RBF | RQ
  isotropy: false                # false = one length scale per parameter
  auto_select: false             # true = fit all three kernels and keep the best (about 3x the time)
  n_restarts: 10                 # restarts of the hyperparameter optimizer
  noisy: true                    # add a white-noise term
  nugget: 1.0e-9                 # jitter on the kernel diagonal
  normalize_x_method: norm       # norm | standard | none
  transform_y_method: norm       # norm | standard | none
  gpe_reg_method: LBFGS
  dim_red_method: 'no'           # 'no' | pca
  input_transform: user
```

Only the keys you set need to be present. When a key is left out:

| Key | Behaviour |
|---|---|
| `type` | `pce` |
| `output_qoi` | `I-129` |
| `z_interest` | required |
| `t_interest` | all time steps |
| `n_init` | all training samples |
| `qoi_floor` | `1e-20` for flux QoIs, `1e-12` for concentration QoIs |
| `transform` | none |
| `parallel` | false |
| `pce_opts` | `type: aPCE`, `reg_method: OLS`, `q_norm: 1.0`, `degree: [1]`, `dim_red_method: 'no'`, `bootstrap_method: fast`, `n_bootstrap_itrs: 1`, `input_transform: user` |
| `gp_opts` | `kernel_type: RBF`, `isotropy: false`, `auto_select: false`, `n_restarts: 10`, `noisy: false`, `nugget: 1e-9`, `normalize_x_method: norm`, `transform_y_method: norm`, `gpe_reg_method: LBFGS`, `dim_red_method: 'no'`, `input_transform: user` |

Each key in a block is looked up on its own, so a partial `pce_opts` or
`gp_opts` block is completed from the defaults above.

`pce_gpe` always uses a noisy GP, and takes the largest value of `pce_opts.degree`.

### `z_interest`

Depths are z coordinates in metres. Each entry can be:

| Entry | Meaning |
|---|---|
| a number, e.g. `-400.0` | that depth |
| a layer boundary, e.g. `Host_rock_top`, `Host_rock_bottom` | the depth of that boundary, read from `geometry/` in the case folder |
| `[top, bottom]` | all mesh nodes between the two depths; each end can be a number or a boundary name |
| `all` | the whole column |
| `{range: [top, bottom]}` | same as `[top, bottom]` |

A depth where the model output does not vary across the samples is dropped.

## Outputs

Everything is written to `--engine_folder`:

| File | Content |
|---|---|
| `engine.pkl` | The trained surrogate (a BayesValidRox `Engine`, saved with joblib). |
| `engine.pkl.manifest.json` | Case, engine name, type, settings, quantity of interest, resolved depths, number of training samples, and the file names of the training and validation data. |
| `validation_metrics.json` | Validation metrics per output depth and time step: `rmse`, `mse`, `nse`, `r2`, `pearson_r`, `mean_error`, `std_error`, `norm_error`, `P95`, `DS`. Time steps with no signal are `NaN`. Only written when validation data is given. |
| `training_summary.txt` | Method, number of training points, training time and the validation metrics averaged over the output locations. |

## Layout of the code

| Path | Role |
|---|---|
| `train_surrogate.py` | Command-line entry point. |
| `src/experiment.py` | Reads the settings and works out the training options. |
| `src/runner.py` | Loads the data, trains, validates and saves. |
| `src/prepare_data.py` | Builds the input space from the uncertain parameters. |
| `src/data_io.py`, `src/scaling.py`, `src/validation.py`, `src/geometry.py`, `src/manifest.py` | Reading the HDF5 results, output scaling, validation metrics, layer depths, manifest files. |
| `src/bayesvalidrox/` | The surrogate library (a local copy). |
