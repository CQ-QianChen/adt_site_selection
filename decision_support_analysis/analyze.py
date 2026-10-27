import argparse
import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import yaml

from nuctrans_fastuq.analysis.io import depth_index, parameter_rows, parameter_units, read_stratigraphy, read_uq
from nuctrans_fastuq.analysis.plots.inputs import inputs, stratigraphy
from nuctrans_fastuq.analysis.plots.propagation import dependence, distribution, profile, timeseries
from nuctrans_fastuq.analysis.plots.sensitivity import sobol_heatmap, sobol_ranking
from nuctrans_fastuq.analysis.stats import key_numbers

PLOTS = {"stratigraphy": stratigraphy, "inputs": inputs, "profile": profile, "timeseries": timeseries, "dependence": dependence, "distribution": distribution, "sobol_heatmap": sobol_heatmap, "sobol_ranking": sobol_ranking}
KEYS = {"qoi", "levels", "labels", "text", "style", "figures"}


def tag(value):
    return "-".join(map(tag, value)) if isinstance(value, list) else f"{value:g}" if isinstance(value, (int, float)) else str(value)


def expand(figures):
    """One figure per depth when depth is a list."""
    out = []
    for item in figures:
        out += [{**item, "depth": depth} for depth in item["depth"]] if isinstance(item.get("depth"), list) else [item]
    return out


def with_times(items, propagation):
    """Plots at one time get the time of the peak of the median (and the last time when the peak is earlier) unless time is given; time: peak is the peak only."""
    out = []
    for item in items:
        function = PLOTS.get(item.get("plot"))
        if function is None or "time" not in inspect.signature(function).parameters or item.get("time") not in (None, "peak"):
            out.append(item)
            continue
        times = propagation["axes/time"]
        peak = np.abs(np.median(propagation["output/mean"][:, :, depth_index(propagation, item["depth"])[0]], axis=0)).argmax()
        out += [{**item, "time": float(times[n])} for n in dict.fromkeys([peak] + ([len(times) - 1] if item.get("time") is None else []))]
    return out


def generate(propagation_path, sensitivity_path, geometry_path, scenario_path, output_dir, data=(), engine=None, scenario_hash=None, uq_hash=None, analysis_hash=None):
    """Write stats.json, the figures in figures/ as listed in the analysis config and figures.json, an index of them for the report."""
    scenario = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8"))
    config = scenario.get("analysis") or {}
    if set(config) - KEYS or not config.get("figures"):
        raise SystemExit(f"analysis config: keys are {sorted(KEYS)} and figures is required, got {sorted(config)}")
    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    propagation, sensitivity, layers = read_uq(propagation_path), read_uq(sensitivity_path), read_stratigraphy(geometry_path)
    qoi = config.get("qoi", {})
    raw_unit = propagation["qoi_unit"]
    factor = qoi.get("factor") or 1  # null or no factor: the raw output
    propagation["output/mean"] = propagation["output/mean"] * factor  # the QoI in the unit of the analysis, the threshold is in this unit too
    propagation["qoi_unit"] = qoi.get("unit", raw_unit)
    text = {"qoi": False, "title": "{time}", **({"quantity": qoi["quantity"]} if "quantity" in qoi else {}), **config.get("text", {})}  # the report names the QoI, the figures do not repeat it
    common = {"text": text, "style": {"font": 10, "width": 130, **config.get("style", {})}}  # one font size and a width that fits the page, figures are placed at their size
    rows, names, index = parameter_rows(sensitivity["axes/parameter_names"], layers), {}, []
    units = parameter_units(sensitivity["axes/parameter_names"], data)
    units = {label: units[sensitivity["axes/parameter_names"][j]] for label, j in rows}
    labels = {depth: name for name, depth in config.get("labels", {}).items()}  # depth (a number or a layer name) -> label
    stats = key_numbers(propagation, sensitivity, rows, labels, qoi.get("threshold"), config.get("levels"))
    stats["qoi"] = " ".join(filter(None, [qoi.get("nuclide"), qoi.get("quantity")])) or propagation["output_qoi"]
    metrics = Path(engine or "") / "validation_metrics.json"
    if metrics.exists():  # R2 of the surrogate against the model at the labelled depths, over the times
        r2 = json.loads(metrics.read_text(encoding="utf-8"))["r2"]
        numeric = {float(key): key for key in r2 if re.fullmatch(r"-?[\d.]+(e-?\d+)?", key)}
        stats["validation"] = {}
        for depth, name in labels.items():
            if str(depth) in r2:  # keyed by the name of the output location
                key = str(depth)
            else:
                z = propagation["axes/depth_z"][depth_index(propagation, depth)[0]]
                key = numeric[min(numeric, key=lambda value: abs(value - z))]
            values = np.array(r2[key], dtype=float)
            stats["validation"][name] = {"r2_min": float(np.nanmin(values)), "r2_mean": float(np.nanmean(values))}
    stats["provenance"] = {"qoi_factor": factor, "raw_unit": raw_unit, "scenario_hash": scenario_hash, "uq_hash": uq_hash, "analysis_hash": analysis_hash, "data": {Path(folder).name: hashlib.sha256(b"".join(f.read_bytes() for f in sorted(Path(folder).rglob("*")) if f.is_file())).hexdigest()[:8] for folder in data}}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=lambda value: value.item()), encoding="utf-8")

    for item in with_times(expand(config["figures"]), propagation):
        item = dict(item)
        name = item.pop("plot", None)
        style = {**common["style"], **item.pop("style", {})}  # a figure can override style options
        if name not in PLOTS:
            raise SystemExit(f"analysis config: unknown plot {name!r}, available: {sorted(PLOTS)}")
        function = PLOTS[name]
        parameters = inspect.signature(function).parameters
        unknown = set(item) - set(parameters)
        if unknown:
            raise SystemExit(f"analysis config: {name} has no option {sorted(unknown)}, options: {[k for k in parameters if k not in ('path', 'text', 'style')][1:]}")
        stem = re.sub(r"[^\w.-]+", "", "_".join([name] + [k + tag(item[k]) for k in ("depth", "times", "time") if k in item]))
        names[stem] = names.get(stem, 0) + 1
        path = out / "figures" / f"{stem}{'' if names[stem] == 1 else '_' + str(names[stem])}.png"
        extra = {"rows": rows} if "rows" in parameters else {}
        if "units" in parameters:
            extra["units"] = units
        if "threshold" in parameters and "threshold" not in item:
            extra["threshold"] = qoi.get("threshold")
        if "marks" in parameters and "marks" not in item:
            extra["marks"] = {k: v for k, v in labels.items() if isinstance(k, (int, float))}
        data = sensitivity if name.startswith("sobol") else propagation
        args = [data] + ([layers] if "layers" in parameters else []) + [path]
        depth = item.get("depth")
        label = labels.get(depth)
        function(*args, **extra, text={**common["text"], **({"where": label} if label else {})}, style=style or None, **item)
        index.append({"plot": name, "file": path.name, "depth": label or depth, "time": item.get("time", item.get("times"))})
        print(f"wrote {path}")
    (out / "figures.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stats and figures of one scenario from its propagation, sensitivity and geometry files.")
    parser.add_argument("--propagation", required=True)
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--geometry", required=True, help="rock_interface.yaml")
    parser.add_argument("--data", nargs="*", default=[], help="folders with the input data (geometry, rock_data, ...), for the units of the inputs")
    parser.add_argument("--engine", help="engine folder, for the validation metrics")
    parser.add_argument("--scenario", required=True, help="yaml of the scenario with its analysis block")
    parser.add_argument("--hash", help="hash of the scenario, for the run information")
    parser.add_argument("--uq-hash", help="hash of the uq results, for the run information")
    parser.add_argument("--analysis-hash", help="hash of the analysis, for the run information")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate(args.propagation, args.sensitivity, args.geometry, args.scenario, args.output_dir, args.data, args.engine, args.hash, args.uq_hash, args.analysis_hash)


if __name__ == "__main__":
    main()
