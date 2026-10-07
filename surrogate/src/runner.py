"""
runner.py - Train a BayesValidRox surrogate on OGS results.
"""
import hickle
import json
import yaml
from pathlib import Path
import warnings
import numpy as np
import time

from . import prepare_data
from .scaling import fit_scaling, scale_data, descale_data, descale_std
from .validation import validation_error

from .data_io import load_and_extract_data

from .bayesvalidrox import (PyLinkForwardModel, ExpDesigns, Engine)


def _get_unique_path(base_path):
    """base_path, or base_path with _v2, _v3, ... if it already exists."""
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_v{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _build_summary_lines(method, n_tp, t_elapsed, val_metrics, output_names):
    """Lines for training_summary.txt."""
    lines = [
        f"Method          : {method}",
        f"Training points : {n_tp}",
        f"Training time   : {t_elapsed:.1f} s",
    ]
    if val_metrics is None:
        lines.append("\nNo validation data provided.")
        return lines

    lines.append("\nValidation metrics (mean over observation locations):")
    header = f"{'Output':<20} {'R2':>8} {'RMSE':>12} {'Pearson-R':>12} {'NSE':>8} {'Mean Err':>12} {'Std Err':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for out in output_names:
        if out not in val_metrics.get('r2', {}):
            continue
        r2        = np.nanmean(val_metrics['r2'][out])
        rmse      = np.nanmean(val_metrics['rmse'][out])
        pearson_r = np.nanmean(val_metrics['pearson_r'][out])
        nse       = np.nanmean(val_metrics['nse'][out])
        # nanmean: early time steps have zero concentration, so these are nan there
        mean_err  = np.nanmean(val_metrics['mean_error'][out])
        std_err   = np.nanmean(val_metrics['std_error'][out])
        lines.append(
            f"{out:<20} {r2:>8.4f} {rmse:>12.4e} {pearson_r:>12.4f} {nse:>8.4f} {mean_err:>12.4e} {std_err:>10.4e}"
        )

    return lines


def _check_outputs_vary(outputs, min_varying_fraction=0.05):
    """Raise if an output varies in too few time steps to be fitted.

    outputs: {label: array(n_samples, n_timesteps)}
    """
    dead = []
    for label, arr in outputs.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a[:, None]
        varying = (a.var(axis=0) > 0).mean() if a.size else 0.0
        if varying <= min_varying_fraction:
            dead.append((label, varying))

    if not dead:
        return

    lines = [
        "These outputs barely vary across the training samples, so a surrogate "
        "cannot be fitted to them:",
    ]
    for label, frac in dead:
        lines.append(f"    {label}: only {frac:.1%} of time steps vary")
    lines += [
        "",
        "The usual cause is a depth pinned by a boundary condition - the source "
        "location is fixed by construction, so changing the parameters does not "
        "move it. Pick depths away from the source, where the response actually "
        "responds to the inputs.",
    ]
    raise ValueError("\n".join(lines))


def train_surrogate(
    meta_model_opts,
    exp_design_opts=None,
    parameters_path="sampling/input/sampling_config.yaml",
    case_dir=None,          # output/{Case}/
    config_path=None,       # model config, optional
    training_path=None,
    validation_path=None,
    output_qoi="I-129",
    z_interest=None,
    t_interest=None,
    n_init=500,
    sampling_method=None,  # 'user' or 'sobol'; None -> 'user' when there is training data
    parallel=False,
    verbose=True,
    transform=None,
    qoi_floor=None,      # numerical noise floor threshold
    z_range=None,        # (top, bottom) depth window or list of intervals
    z_min_variance=None, # drop depths that do not vary across samples
    engine_path=None,    # where to save engine.hkl (Experiment.engine_path)
):
    """Load the data, set up the input space and train the surrogate.

    Returns the trained BayesValidRox Engine.
    """
    parameters_path = Path(parameters_path)
    if case_dir is None:
        raise ValueError(
            "case_dir is required: pass output/{Case}/, the folder holding "
            "rock_data/, geometry/ and the nuclide_*_data/ folders."
        )
    case_dir = Path(case_dir)

    # 1. Load the sampling config (which parameters are uncertain)
    with open(parameters_path, "r") as f:
        uncertain_parameters = yaml.safe_load(f)

    # Parameter order from the config; X columns and marginals both follow it
    param_names = prepare_data.get_param_names_from_config(uncertain_parameters)

    # 2. Load training and validation data
    if training_path is None or not Path(training_path).exists():
        raise FileNotFoundError(
            f"Training data path is required and must exist. Received: {training_path}"
        )

    X_train, train_output_at_x, _, t_values_array = load_and_extract_data(
        training_path, output_qoi, z_interest, t_interest,
        param_names=param_names, qoi_floor=qoi_floor,
        z_range=z_range, z_min_variance=z_min_variance,
    )

    if train_output_at_x is not None:
        # If n_init caps the training samples, check variance on the actual training slice
        var_thresh = z_min_variance if z_min_variance is not None else 1e-30
        if n_init is not None and X_train is not None and n_init < X_train.shape[0]:
            flat_keys = [
                k for k, v in train_output_at_x.items()
                if np.asarray(v[:n_init]).var(axis=0).max() <= var_thresh
            ]
            if flat_keys:
                print(f"  z_interest: dropped {len(flat_keys)} depths with variance <= {var_thresh:g} "
                      f"in the first {n_init} training samples.")
                for k in flat_keys:
                    del train_output_at_x[k]

        _check_outputs_vary(train_output_at_x)

    # Output scaling, if a transform is set
    scaling_config = None
    if transform is not None and transform != 'none':
        scaling_config = fit_scaling(train_output_at_x, scale_type=transform)
        if train_output_at_x is not None:
            train_output_at_x = scale_data(train_output_at_x, scaling_config)
        
    X_val, valid_y = None, None
    if validation_path is not None:
        if Path(validation_path).exists():
            # Keep the same depths as training
            X_val, valid_y, _, _ = load_and_extract_data(
                validation_path, output_qoi, z_interest, t_interest,
                param_names=param_names, qoi_floor=qoi_floor,
                keep_keys=(list(train_output_at_x)
                           if train_output_at_x is not None else None),
                z_range=z_range, z_min_variance=z_min_variance,
            )
            if valid_y is not None and scaling_config is not None:
                valid_y = scale_data(valid_y, scaling_config)
        else:
            warnings.warn(
                f"Validation data path {validation_path} does not exist. Running with no validation.",
                UserWarning
            )

    # 3. Forward model. Never called, but BVR's Engine needs one.
    model = PyLinkForwardModel()
    model.link_type = "Function"
    model.py_file = "runner"
    model.name = "OGSModel"

    # Output names
    if X_train is not None:
        output_names = list(train_output_at_x.keys())
    else:
        output_names = prepare_data.get_output_names(z_interest, t_interest, output_qoi)

    model.output.names = output_names
    model.func_args = {
        "uncertain_parameters": uncertain_parameters,
        "param_names": param_names,
        "z_interest": z_interest,
        "t_interest": t_interest,
        "default_qoi": output_qoi,
    }

    # 4. Input space (marginals), from the case's data folders
    inputs = prepare_data.create_input_space(uncertain_parameters, case_dir)

    # Give the inputs to the metamodel
    if callable(meta_model_opts):
        meta_model_opts = meta_model_opts(inputs)
    else:
        meta_model_opts.input_obj = inputs

    # 5. Experimental design
    if callable(exp_design_opts):
        exp_design = exp_design_opts(inputs)
    elif exp_design_opts is not None:
        exp_design = exp_design_opts
        exp_design.input_object = inputs
    else:
        if sampling_method is None:
            sampling_method = 'user' if X_train is not None else 'sobol'
            
        n_init_samples = n_init if n_init is not None else (X_train.shape[0] if X_train is not None else 1)
        exp_design = ExpDesigns(
            input_object=inputs,
            sampling_method=sampling_method,
            n_init_samples=n_init_samples
        )
        
    # Put the training and validation data into exp_design
    if X_train is not None:
        if hasattr(exp_design, 'n_init_samples') and exp_design.n_init_samples is not None:
            n_init = exp_design.n_init_samples
            
        num_tp = X_train.shape[0]
        if n_init is None:
            n_init = num_tp
        elif num_tp < n_init:
            warnings.warn(
                f"Number of training points provided ({num_tp}) is smaller than n_init ({n_init}). "
                f"{num_tp} points will be used.",
                UserWarning
            )
            n_init = num_tp

        exp_design.n_init_samples = n_init
        exp_design.n_max_samples = n_init
        exp_design.x = X_train[0:n_init, :]
        exp_design.y = {}
        for key in model.output.names:
            exp_design.y[key] = train_output_at_x[key][0:n_init, :]
        exp_design.y['x_values'] = t_values_array

    if X_val is not None:
        exp_design.x_valid = X_val
        exp_design.y_valid = valid_y

    print(f'Training surrogate with {exp_design.x.shape[0]} points.')

    # 6. Train
    engine = Engine(meta_model_opts, model, exp_design)
    engine.scaling_config = scaling_config
    t_start = time.time()
    engine.train_normal(parallel=parallel, verbose=verbose)
    t_elapsed = time.time() - t_start

    # 7. Save
    if engine_path is None:
        raise ValueError("engine_path is required: provide the destination path for engine.hkl.")
    engine_file = Path(engine_path)
    output_path = engine_file.parent
    output_path.mkdir(parents=True, exist_ok=True)
    # TODO: hickle currently stores the Engine as one pickled blob inside the HDF5 file, so the
    # contents are opaque. Save plain arrays/dicts (PCE coefficients, multi-indices, input specs) instead.
    hickle.dump(engine, engine_file, mode='w', compression='gzip', compression_opts=4)

    # 8. Validation metrics
    val_metrics = None
    if X_val is not None:
        y_pred, y_std = engine.eval_metamodel(X_val)
        # Back to physical units
        if scaling_config is not None:
            y_pred_phys = descale_data(y_pred, scaling_config)
            y_std_phys = descale_std(y_std, scaling_config)
            valid_y_phys = descale_data(valid_y, scaling_config)
        else:
            y_pred_phys = y_pred
            y_std_phys = y_std
            valid_y_phys = valid_y
        val_metrics = validation_error(true_y=valid_y_phys, sim_y=y_pred_phys, sim_std=y_std_phys, std_metrics=True)
        with open(output_path / "validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2, default=lambda o: np.asarray(o).tolist())

    # 9. training_summary.txt (GPs have no pce_reg_method, so show the kernel)
    _reg_str     = (getattr(engine.meta_model, 'pce_reg_method', '')
                    or str(getattr(engine.meta_model, '_kernel_type', '') or ''))
    method_str   = f"{getattr(engine.meta_model, 'meta_model_type', 'PCE')} / {_reg_str}"
    n_tp         = engine.exp_design.x.shape[0]
    lines        = _build_summary_lines(method_str, n_tp, t_elapsed, val_metrics, engine.out_names)
    if verbose:
        print("\n".join(lines))
    summary_path = _get_unique_path(output_path / "training_summary.txt")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if verbose:
        print(f"Metamodel engine trained and saved to {engine_file}")
        
    return engine
