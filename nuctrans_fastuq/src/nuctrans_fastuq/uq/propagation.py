"""
Uncertainty propagation through a saved engine: draw input samples, evaluate the surrogate
and write inputs, predicted mean and predictive std to one HDF5 file (axes under `axes/`,
a `dims` attribute on every dataset).

    python -m nuctrans_fastuq.uq.propagation --engine engine.h5 --config propagation_opts.yaml --output out.h5

Settings: n_samples and seed (required), sampling_method (default random). Any failure stops
the task with the error and writes no file.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from ..surrogate.engine_io import load_engine
from .sensitivity import _STR, _dataset, _to_float

KEYS = ("n_samples", "seed")
SAMPLING_METHODS = ("random", "latin_hypercube", "sobol", "halton", "hammersley")


def check_settings(settings):
    """The settings of the task: n_samples (> 0) and seed, both required, sampling_method optional, nothing else."""
    settings = dict(settings or {})
    missing = [k for k in KEYS if k not in settings]
    if missing:
        raise ValueError(f"missing propagation settings: {missing}")
    unknown = sorted(set(settings) - set(KEYS) - {"sampling_method"})
    if unknown:
        raise ValueError(f"unknown propagation settings: {unknown}")
    n, seed = settings["n_samples"], settings["seed"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    method = settings.get("sampling_method", "random")
    if method not in SAMPLING_METHODS:
        raise ValueError(f"sampling_method must be one of {list(SAMPLING_METHODS)}, got {method!r}")
    return {"n_samples": n, "seed": seed, "sampling_method": method}


def propagate(engine, n_samples, seed, sampling_method="random"):
    """Sample the inputs and evaluate the surrogate: X [sample, param], mean and std [sample, time, depth]."""
    np.random.seed(seed)  # chaospy draws from the global numpy generator
    x = np.asarray(engine.exp_design.generate_samples(n_samples, sampling_method), dtype=float)
    mean, std = engine.meta_model.eval_metamodel(x)  # per depth: [sample, time]
    stack = lambda res: np.stack([np.asarray(res[d], dtype=float) for d in engine.out_names], axis=-1)  # noqa: E731
    return x, stack(mean), stack(std)


def write_propagation(engine, path, header, settings):
    """Propagate the inputs through the engine and write path (kept untouched if this fails)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        settings = check_settings(settings)
        x, mean, std = propagate(engine, **settings)
        with h5py.File(tmp, "w") as f:
            for key, value in header.items():
                f.attrs[key] = value
            f.attrs["n_samples"] = settings["n_samples"]
            f.attrs["seed"] = settings["seed"]
            f.attrs["sampling_method"] = settings["sampling_method"]
            f.attrs["settings_json"] = json.dumps(settings)
            f.attrs["created"] = datetime.now().isoformat()

            axes = f.create_group("axes")
            axes.create_dataset("parameter_names", data=np.array(engine.meta_model.par_names, dtype=object), dtype=_STR)
            axes.create_dataset("time", data=np.asarray(engine.exp_design.x_values, dtype=float))
            axes.create_dataset("depth", data=np.array(engine.out_names, dtype=object), dtype=_STR)
            axes.create_dataset("depth_z", data=np.array([_to_float(d) for d in engine.out_names]))

            _dataset(f.create_group("input"), "X", x, ["sample", "param"])
            out = f.create_group("output")
            for name, data in (("mean", mean), ("std", std)):
                _dataset(out, name, data, ["sample", "time", "depth"], compression="gzip", compression_opts=4)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Propagate the input uncertainty through a saved engine.")
    parser.add_argument("--engine", required=True, help="engine.h5 written by the surrogate task.")
    parser.add_argument("--config", required=True, help="yaml with n_samples and seed.")
    parser.add_argument("--output", required=True, help="HDF5 file to write.")
    args = parser.parse_args()

    with h5py.File(args.engine, "r") as f:
        header = {k: str(f.attrs[k]) for k in ("scenario", "engine_name", "surrogate_type", "output_qoi")}
    with open(args.config, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    write_propagation(load_engine(args.engine), args.output, header, settings)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
