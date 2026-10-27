import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from fpdf.fonts import FontFace
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from build import HAIRLINE, INK, MUTED, PREDICTION, SUMMARY, Report, drivers, sci

matplotlib.use("Agg")
COLORS = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#8C8C8C"]  # one per site, none of them the red of the threshold
THRESHOLD = "#B94A4A"
FLOOR = 1e-3  # the plots start this fraction below the smallest 95th percentile peak
ENGLISH = {"bavaria": "bayern", "hesse": "hessen", "lowersaxony": "niedersachsen", "mecklenburgwesternpomerania": "mecklenburgvorpommern", "northrhinewestphalia": "nordrheinwestfalen",
           "rhinelandpalatinate": "rheinlandpfalz", "saxony": "sachsen", "saxonyanhalt": "sachsenanhalt", "thuringia": "thueringen"}  # the states with another English name


def simple(name):
    return re.sub(r"[^a-z]", "", name.lower().replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("ß", "ss"))


def style(width, font=None):
    """Figure size in inches, line width factor and rc parameters: the same look as the figures of the single reports."""
    font = font or max(10, 22 * width / 25.4 / 12)
    k = font / 22
    return k, {"font.size": font, "axes.titlesize": 1.1 * font, "axes.labelsize": font, "xtick.labelsize": 0.9 * font, "ytick.labelsize": 0.9 * font, "legend.fontsize": 0.9 * font,
               "axes.linewidth": 0.8 * k, "lines.linewidth": 1.5 * k, "xtick.major.width": 0.8 * k, "ytick.major.width": 0.8 * k, "xtick.minor.width": 0.6 * k, "ytick.minor.width": 0.6 * k,
               "xtick.major.size": 3.5 * k, "ytick.major.size": 3.5 * k, "xtick.minor.size": 2 * k, "ytick.minor.size": 2 * k, "xtick.major.pad": 3.5 * k, "ytick.major.pad": 3.5 * k,
               "axes.labelpad": 4 * k, "axes.titlepad": 6 * k, "axes.formatter.use_mathtext": True}


def curves(runs):
    """5th percentile, median and 95th percentile of the magnitude over time of each run, the lowest value and the first time that the plots show."""
    out = [(run["label"], run["time"], np.percentile(np.abs(run["y"]), [5, 50, 95], axis=0)) for run in runs]
    floor = min(band[2].max() for *_, band in out) * FLOOR
    return out, floor, min(time[band[2] >= floor][0] for _, time, band in out)


def timeseries(runs, path, quantity, unit, threshold):
    """Median and band of all runs on shared log axes."""
    k, rc = style(160)
    bands, floor, start = curves(runs)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(160 / 25.4, 0.58 * 160 / 25.4))
        for (label, time, (low, median, high)), color in zip(bands, COLORS):
            ax.fill_between(time[time > 0], low[time > 0], high[time > 0], color=color, alpha=0.2, lw=0)
            ax.plot(time[time > 0], median[time > 0], color=color, lw=3 * k, label=label)
        if threshold is not None:
            ax.axhline(threshold, color=THRESHOLD, lw=2 * k, ls=":", label="threshold")
        ax.set(xlabel="Time [yr]", ylabel=f"|{quantity}| [{unit}]", xscale="log", yscale="log", xlim=(start, None), ylim=(floor, None))
        ax.grid(alpha=0.3, lw=0.8 * k)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


def site_map(runs, regions, path, quantity, unit, threshold):
    """A map of the regions with one marker per run, linked to its time series (north to south, shared axes)."""
    k, rc = style(180, 9)

    def project(lon, lat, lon0=10.45, lat0=51.16, p1=48.67, p2=54.0):  # Lambert conformal conic
        lon, lat, lon0, lat0, p1, p2 = (np.radians(v) for v in (lon, lat, lon0, lat0, p1, p2))
        n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
        f = np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** n / n
        rho, rho0 = f / np.tan(np.pi / 4 + lat / 2) ** n, f / np.tan(np.pi / 4 + lat0 / 2) ** n
        return rho * np.sin(n * (lon - lon0)), rho0 - rho * np.cos(n * (lon - lon0))

    bands, floor, start = curves(runs)
    sites = []
    for (label, time, band), run, color in zip(bands, runs, COLORS):
        x, y = max(regions[run["region"]], key=len).T  # area centroid of the main ring
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        marker = project(((x + np.roll(x, -1)) * cross).sum() / (3 * cross.sum()), ((y + np.roll(y, -1)) * cross).sum() / (3 * cross.sum()))
        sites.append({"label": label, "region": run["region"], "color": color, "marker": marker, "time": time, "band": band})
    sites.sort(key=lambda site: -site["marker"][1])

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(180 / 25.4, 0.62 * 180 / 25.4))
        ax = fig.add_axes([0, 0.02, 0.42, 0.96])
        ax.set_aspect("equal")
        ax.axis("off")
        tints = {site["region"]: site["color"] for site in sites}
        for name, rings in regions.items():
            for ring in rings:
                ax.fill(*project(ring[:, 0], ring[:, 1]), facecolor=tints.get(name, "#ECECEC"), alpha=0.35 if name in tints else 1, edgecolor="white", lw=0.6 * k)
        height, panels = 0.84 / len(sites), []
        for n, site in enumerate(sites):
            panel = fig.add_axes([0.56, 0.92 - (n + 1) * height + 0.14 * height, 0.43, 0.68 * height], sharex=panels[0] if panels else None, sharey=panels[0] if panels else None)
            time, (low, median, high) = site["time"], site["band"]
            panel.fill_between(time[time > 0], low[time > 0], high[time > 0], color=site["color"], alpha=0.2, lw=0)
            panel.plot(time[time > 0], median[time > 0], color=site["color"], lw=3 * k)
            if threshold is not None:
                panel.axhline(threshold, color=THRESHOLD, lw=1.5 * k, ls=":")
            panel.set(xscale="log", yscale="log", xlim=(start, None), ylim=(floor, None))
            panel.set_title(f"{chr(ord('a') + n)}   {site['label']}", loc="left", fontweight="bold")
            panel.grid(alpha=0.3, lw=0.8 * k)
            panel.spines[["top", "right"]].set_visible(False)
            panel.xaxis.set_minor_formatter(NullFormatter())
            panel.tick_params(labelbottom=n == len(sites) - 1)
            panels.append(panel)
        panels[-1].set_xlabel("Time [yr]")
        fig.text(0.99, 0.995, f"|{quantity}| [{unit}]", ha="right", va="top")
        fig.canvas.draw()  # the map is final only after a draw
        for n, (site, panel) in enumerate(zip(sites, panels)):
            mx, my = site["marker"]
            ax.plot(mx, my, "o", color=site["color"], ms=7 * k, mec="white", mew=1.2 * k, zorder=5)
            ax.annotate(chr(ord("a") + n), (mx, my), xytext=(0, 9 * k), textcoords="offset points", ha="center", fontweight="bold", color=site["color"])
            a, b = fig.transFigure.inverted().transform(ax.transData.transform((mx, my))), panel.get_position()
            fig.add_artist(Line2D([a[0], b.x0 - 0.075], [a[1], (b.y0 + b.y1) / 2], transform=fig.transFigure, color=site["color"], lw=0.8 * k, alpha=0.7))
        fig.savefig(path, dpi=300)
        plt.close(fig)


def table(pdf, widths, heads, rows, line_height=6.5):
    pdf.set_draw_color(*HAIRLINE)
    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "", 8.5)
    with pdf.table(borders_layout="HORIZONTAL_LINES", col_widths=widths, line_height=line_height, text_align=("LEFT",) * len(widths), cell_fill_color=(255, 255, 255), cell_fill_mode="ALL",
                   headings_style=FontFace(emphasis="BOLD", size_pt=7, color=MUTED, fill_color=(255, 255, 255))) as t:
        t.row(heads)
        for row in rows:
            t.row(row)


def main():
    parser = argparse.ArgumentParser(description="PDF report comparing the analysis folders of several runs.")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=FOLDER", help="label and analysis folder (in its uq folder) of one run, once per run")
    parser.add_argument("--regions", help="geojson of the regions (states) for the map; the region of a run is communication_sdh.tag_dict.location in its scenario")
    parser.add_argument("--title", default="Site comparison")
    parser.add_argument("--output-dir", required=True, help="gets comparison.pdf and the figures")
    args = parser.parse_args()
    runs = []
    for spec in args.run:
        label, folder = spec.split("=", 1)
        folder = Path(folder)
        scenario = yaml.safe_load((folder / "scenario.yaml").read_text(encoding="utf-8"))
        stats = json.loads((folder / "stats.json").read_text())
        with h5py.File(folder.parent.parent / "propagation.h5") as f:  # the uq folder, two levels up in the datastore layout
            output, time, names, z = f["output/mean"][()] * stats["provenance"]["qoi_factor"], f["axes/time"][()], list(f["axes/depth"].asstr()[()]), f["axes/depth_z"][()]  # in the unit of the analysis
        region = simple(scenario["communication_sdh"]["tag_dict"]["location"][0])
        depths = {}  # location -> output over samples and time at its depth (a layer name or a depth in m)
        for location, depth in scenario["analysis"].get("labels", {}).items():
            depths[location] = output[:, :, names.index(depth) if isinstance(depth, str) else int(np.abs(z - depth).argmin())]
        runs.append({"label": label, "stats": stats, "time": time, "depths": depths, "region": ENGLISH.get(region, region)})

    first = runs[0]["stats"]
    headline, unit = first["headline"], first["header"]["qoi_unit"]
    thresholds = lambda stats: {k: v.get("threshold", {}).get("value") for k, v in stats["depths"].items()}
    for run in runs[1:]:  # the runs must measure the same thing
        for name, a, b in (("quantity", first["qoi"], run["stats"]["qoi"]), ("unit", unit, run["stats"]["header"]["qoi_unit"]), ("time span", first["period"], run["stats"]["period"]),
                           ("locations", list(first["depths"]), list(run["stats"]["depths"])), ("thresholds", thresholds(first), thresholds(run["stats"]))):
            if a != b:
                raise SystemExit(f"{runs[0]['label']} and {run['label']} differ in {name}: {a} and {b}")

    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    figures = {}
    for location, threshold in thresholds(first).items():
        figures[location] = Path("figures") / f"timeseries_{re.sub(r'[^a-z0-9]+', '_', location.lower())}.png"
        timeseries([{**run, "y": run["depths"][location]} for run in runs], out / figures[location], first["qoi"], unit, threshold)
    if args.regions:
        regions = {}
        for feature in json.loads(Path(args.regions).read_text(encoding="utf-8"))["features"]:
            g = feature["geometry"]
            regions[simple(feature["properties"]["name"])] = [np.array(p[0], dtype=float) for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"])]
        site_map([{**run, "y": run["depths"][headline]} for run in runs], regions, out / "figures" / "map.png", first["qoi"], unit, thresholds(first)[headline])

    threshold = thresholds(first)[headline]
    ranked = sorted(runs, key=lambda run: (-run["stats"]["depths"][headline].get("threshold", {}).get("share", 0), -abs(run["stats"]["depths"][headline]["peak"]["median"])))
    pdf = Report(args.title)
    pdf.add_page()
    pdf.opener("SUMMARY", SUMMARY, args.title, f"{len(runs)} sites compared on the {first['qoi']} in {unit}" + (f", against the threshold {sci(threshold, 1)} {unit}" if threshold else "")
               + ". Sites are ordered by " + ("the share of samples above the threshold" if threshold else "the median peak") + f" at {headline.lower()}.")
    if args.regions:
        pdf.image_at(out, {"file": "figures/map.png"})
    for location in first["depths"]:
        pdf.subheading(location)
        rows = []
        for run in ranked:
            d, t = run["stats"]["depths"][location], run["stats"]["depths"][location].get("threshold")
            top = next(iter(drivers(d)), None)
            r2 = run["stats"].get("validation", {}).get(location)
            rows.append([run["label"], sci(d["peak"]["median"]), sci(d["peak"]["high"])]
                        + ([f"{t['share'] * 100:.0f} %", "never" if t["p95_exceeds"] is None else f"{sci(t['p95_exceeds'], 0)} yr"] if threshold else [])
                        + [f"{r2['r2_min']:.3f}" if r2 else "-", top["label"] if top else "-"])
        if threshold:
            table(pdf, (30, 24, 24, 24, 26, 18, 34), ["Site", "Median peak", "95th percentile", "Above threshold", "P95 above from", "R2 min", "Most sensitive"], rows)
        else:
            table(pdf, (30, 30, 30, 24, 66), ["Site", "Median peak", "95th percentile", "R2 min", "Most sensitive"], rows)
        pdf.ln(2)
    pdf.subheading("Runs", "The runs behind this comparison.")
    table(pdf, (30, 50, 25, 25, 25, 25), ["Site", "Scenario", "Scenario hash", "UQ hash", "Analysis hash", "Conversion"],
          [[run["label"], run["stats"]["header"]["scenario"], p["scenario_hash"], p["uq_hash"], p["analysis_hash"], f"x {p['qoi_factor']:g}" if p["qoi_factor"] != 1 else "none"] for run in ranked for p in [run["stats"]["provenance"]]], 6)
    pdf.add_page()
    pdf.opener("PREDICTION", PREDICTION, "Predicted flux over time", "Median (line) and 5-95 % band of the sites on shared axes, as magnitudes" + ("; the dotted red line is the threshold." if threshold else "."))
    for location, path in figures.items():
        pdf.subheading(location)
        pdf.image_at(out, {"file": path})
    pdf.output(out / "comparison.pdf")


if __name__ == "__main__":
    main()
