import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from nuctrans_fastuq.analysis.io import read_uq
from nuctrans_fastuq.analysis.plots.comparison import site_map, timeseries

ENGLISH = {"bavaria": "bayern", "hesse": "hessen", "lowersaxony": "niedersachsen", "mecklenburgwesternpomerania": "mecklenburgvorpommern", "northrhinewestphalia": "nordrheinwestfalen",
           "rhinelandpalatinate": "rheinlandpfalz", "saxony": "sachsen", "saxonyanhalt": "sachsenanhalt", "thuringia": "thueringen"}  # the states with another English name


def simple(name):
    return re.sub(r"[^a-z]", "", name.lower().replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("ß", "ss"))


def main():
    parser = argparse.ArgumentParser(description="Figures and the data of the report comparing the analysis folders of several runs.")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=FOLDER", help="label and analysis folder (in its uq folder) of one run, once per run")
    parser.add_argument("--regions", help="geojson of the regions (states) for the map; the region of a run is communication_sdh.tag_dict.location in its scenario")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    runs = []
    for spec in args.run:
        label, folder = spec.split("=", 1)
        folder = Path(folder)
        scenario = yaml.safe_load((folder / "scenario.yaml").read_text(encoding="utf-8"))
        stats, propagation = json.loads((folder / "stats.json").read_text()), read_uq(folder.parent.parent / "propagation.h5")
        propagation["output/mean"] = propagation["output/mean"] * stats["provenance"]["qoi_factor"]  # in the unit of the analysis, as the stats
        propagation["qoi_unit"] = stats["header"]["qoi_unit"]
        region = simple(scenario["communication_sdh"]["tag_dict"]["location"][0])
        runs.append({"label": label, "stats": stats, "labels": scenario["analysis"].get("labels", {}), "propagation": propagation, "region": ENGLISH.get(region, region)})

    first = runs[0]["stats"]
    thresholds = lambda stats: {k: v.get("threshold", {}).get("value") for k, v in stats["depths"].items()}
    for run in runs[1:]:  # the runs must measure the same thing
        for name, a, b in (("quantity", first["qoi"], run["stats"]["qoi"]), ("unit", first["header"]["qoi_unit"], run["stats"]["header"]["qoi_unit"]), ("time span", first["period"], run["stats"]["period"]),
                           ("locations", list(first["depths"]), list(run["stats"]["depths"])), ("thresholds", thresholds(first), thresholds(run["stats"]))):
            if a != b:
                raise SystemExit(f"{runs[0]['label']} and {run['label']} differ in {name}: {a} and {b}")

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    text, figures = {"quantity": first["qoi"]}, []
    for location, threshold in thresholds(first).items():
        file = f"figures/timeseries_{re.sub(r'[^a-z0-9]+', '_', location.lower())}.png"
        timeseries([(run["label"], run["propagation"], run["labels"][location]) for run in runs], out / file, threshold, text)
        figures.append({"plot": "timeseries", "file": file, "location": location})
    if args.regions:
        regions = {}
        for feature in json.loads(Path(args.regions).read_text(encoding="utf-8"))["features"]:
            g = feature["geometry"]
            regions[simple(feature["properties"]["name"])] = [np.array(p[0], dtype=float) for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"])]
        headline = first["headline"]
        site_map([(run["label"], run["propagation"], run["labels"][headline], run["region"]) for run in runs], regions, out / "figures" / "map.png", thresholds(first)[headline], text, {"font": 9, "width": 180})
        figures.append({"plot": "map", "file": "figures/map.png", "location": headline})
    (out / "comparison.json").write_text(json.dumps({"runs": [{"label": run["label"], "stats": run["stats"]} for run in runs], "figures": figures}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
