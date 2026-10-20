"""
Sobol sensitivity indices of a saved PCE engine: S1 and ST per parameter, time and depth
(mean over the bootstrap replicates, plus the replicates) in one HDF5 file (axes under
`axes/`, a `dims` attribute on every dataset). Indices are NaN where the output has no variance.

    python -m nuctrans_fastuq.uq.sensitivity --engine engine.h5 --config sensitivity_opts.yaml --output out.h5

The indices come from the PCE coefficients, so the config has no keys yet and any key is an
error. Any failure stops the task with the error and writes no file.
"""
import argparse
import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

from ..surrogate.engine_io import load_engine

METHOD = "pce"
_STR = h5py.string_dtype(encoding="utf-8")


def read_header(engine_path):
    with h5py.File(engine_path, "r") as f:
        header = {k: str(f.attrs[k]) for k in ("scenario", "engine_name", "surrogate_type", "output_qoi")}
    header["qoi_unit"] = "mol/(m² year)" if "flux" in header["output_qoi"].lower() else "mol/L"
    return header


def sobol_replicates(engine):
    """S1 and ST of every bootstrap replicate: two arrays [b, param, time, depth]."""
    meta = engine.meta_model
    res = meta.calculate_sobol()
    stack = lambda kind: np.stack([  # noqa: E731
        np.stack([res[depth][kind][p] for p in meta.par_names], axis=1)  # [b, param, time]
        for depth in engine.out_names], axis=-1)
    return stack("S1"), stack("ST")


def _dataset(f, name, data, dims, **kwargs):
    ds = f.create_dataset(name, data=data, **kwargs)
    ds.attrs.create("dims", data=dims, dtype=_STR)
    return ds


def check_settings(settings):
    """The settings of the task; the PCE analysis takes none."""
    if settings:
        raise ValueError(f"unknown sensitivity settings for a PCE engine: {sorted(settings)}")
    return {}


def write_sensitivity(engine, path, header, settings=None):
    """Compute the indices of the engine and write them to path (kept untouched if this fails)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        settings = check_settings(settings)
        s1, st = sobol_replicates(engine)
        meta = engine.meta_model
        with h5py.File(tmp, "w") as f:
            for key, value in header.items():
                f.attrs[key] = value
            f.attrs["method"] = METHOD
            f.attrs["settings_json"] = json.dumps(settings)
            f.attrs["n_bootstrap"] = s1.shape[0]
            f.attrs["created"] = datetime.now().isoformat()

            axes = f.create_group("axes")
            axes.create_dataset("parameter_names", data=np.array(meta.par_names, dtype=object), dtype=_STR)
            axes.create_dataset("time", data=np.asarray(engine.exp_design.x_values, dtype=float))
            axes.create_dataset("depth", data=np.array(engine.out_names, dtype=object), dtype=_STR)
            axes.create_dataset("depth_z", data=np.array([_to_float(d) for d in engine.out_names]))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices stay NaN
                mean = {"S1": np.nanmean(s1, axis=0), "ST": np.nanmean(st, axis=0)}
            dims = ["param", "time", "depth"]
            for kind, data in mean.items():
                _dataset(f, kind, data, dims)
            boot = f.create_group("bootstrap")
            for kind, data in (("S1", s1), ("ST", st)):
                _dataset(boot, kind, data, ["bootstrap"] + dims, compression="gzip", compression_opts=4)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _to_float(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Sobol indices of a saved engine.")
    parser.add_argument("--engine", required=True, help="engine.h5 written by the surrogate task.")
    parser.add_argument("--config", required=True, help="yaml with the sensitivity settings (may be empty).")
    parser.add_argument("--output", required=True, help="HDF5 file to write.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    write_sensitivity(load_engine(args.engine), args.output, read_header(args.engine), settings)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
