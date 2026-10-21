import argparse
import json
from pathlib import Path

from fpdf.fonts import FontFace

from build import HAIRLINE, INK, MUTED, PREDICTION, SUMMARY, Report, drivers, sci


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
    parser = argparse.ArgumentParser(description="PDF report of a comparison from its folder.")
    parser.add_argument("--comparison", required=True, help="folder with comparison.json and the figures")
    parser.add_argument("--title", default="Site comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = Path(args.comparison)
    comparison = json.loads((folder / "comparison.json").read_text())
    runs = [(run["label"], run["stats"]) for run in comparison["runs"]]
    first = runs[0][1]
    headline, unit = first["headline"], first["header"]["qoi_unit"]
    threshold = first["depths"][headline].get("threshold", {}).get("value")
    ranked = sorted(runs, key=lambda run: (-run[1]["depths"][headline].get("threshold", {}).get("share", 0), -abs(run[1]["depths"][headline]["peak"]["median"])))

    pdf = Report(args.title)
    pdf.add_page()
    pdf.opener("SUMMARY", SUMMARY, args.title, f"{len(runs)} sites compared on the {first['qoi']} in {unit}" + (f", against the threshold {sci(threshold, 1)} {unit}" if threshold else "")
               + f". Sites are ordered by " + ("the share of samples above the threshold" if threshold else "the median peak") + f" at {headline.lower()}.")
    for entry in comparison["figures"]:
        if entry["plot"] == "map":
            pdf.image_at(folder, entry)
    for location in first["depths"]:
        pdf.subheading(location)
        rows = []
        for label, stats in ranked:
            d, t = stats["depths"][location], stats["depths"][location].get("threshold")
            top = next(iter(drivers(d)), None)
            rows.append([label, sci(d["peak"]["median"]), sci(d["peak"]["high"]), f"{t['share'] * 100:.0f} %" if t else "-", ("never" if t["p95_exceeds"] is None else f"{sci(t['p95_exceeds'], 0)} yr") if t else "-",
                         f"{stats['validation'][location]['r2_min']:.3f}" if location in stats.get("validation", {}) else "-", top["label"] if top else "-"])
        table(pdf, (30, 24, 24, 24, 26, 18, 34), ["Site", "Median peak", "95th percentile", "Above threshold", "P95 above from", "R2 min", "Most sensitive"], rows)
        pdf.ln(2)
    pdf.subheading("Runs", "The runs behind this comparison.")
    table(pdf, (30, 50, 25, 25, 25, 25), ["Site", "Scenario", "Scenario hash", "UQ hash", "Analysis hash", "Conversion"],
          [[label, stats["header"]["scenario"], p["scenario_hash"], p["uq_hash"], p["analysis_hash"], f"x {p['qoi_factor']:g}" if p["qoi_factor"] != 1 else "none"] for label, stats in ranked for p in [stats["provenance"]]], 6)

    entries = [entry for entry in comparison["figures"] if entry["plot"] == "timeseries"]
    if entries:
        pdf.add_page()
        pdf.opener("PREDICTION", PREDICTION, "Predicted flux over time", "Median (line) and 5-95 % band of the sites on shared axes, as magnitudes; the dotted red line is the threshold.")
        for entry in entries:
            pdf.subheading(entry["location"])
            pdf.image_at(folder, entry)
    pdf.output(args.output)


if __name__ == "__main__":
    main()
