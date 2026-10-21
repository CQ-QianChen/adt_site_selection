import argparse
import json
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace
import yaml
from PIL import Image

INK, MUTED, HAIRLINE = (17, 17, 17), (110, 110, 115), (210, 210, 215)
LEVEL = {"low": (58, 140, 87), "medium": (200, 184, 55), "high": (185, 74, 74)}  # exceedance level -> colour (red as the threshold line in the figures)
INFORMATION, SUMMARY, RELIABILITY, PREDICTION = (253, 224, 227), (232, 232, 237), (211, 243, 200), (212, 235, 251)  # the section colours of the ACTION Twin figure
SECTIONS = [  # (chip, colour, title, plot types): a section without figures is skipped
    ("RELIABILITY ASSESSMENT", RELIABILITY, "Uncertainty quantification", ["profile", "timeseries", "distribution"]),
    ("RELIABILITY ASSESSMENT", RELIABILITY, "Sensitivity analysis", ["sobol_heatmap", "sobol_ranking", "dependence"]),
]
SKIP = {"sim_setup", "get_field_component_index", "process_variables", "secondary_variables", "quantity_of_interests", "create_meshes", "create_python_bc", "mesh_dimension", "project_name", "n_init", "seed"}
LEFT, RIGHT, TOP, BOTTOM = 15, 195, 28, 18  # page margins (mm)


def sci(value, digits=2):
    """1.94e-4"""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}" if value else "0"


def read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def nested(block, *keys):
    for key in keys:
        block = block.get(key) if isinstance(block, dict) else None
    return block


class Report(FPDF):
    def __init__(self, running_head):
        super().__init__(format="A4")
        self.running_head, self.generated = running_head, datetime.now().strftime("%d %b %Y, %H:%M")
        self.set_margins(LEFT, TOP, LEFT)
        self.set_auto_page_break(True, BOTTOM)
        self.alias_nb_pages()

    def header(self):
        self.set_y(9)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.set_draw_color(*HAIRLINE)
        self.set_line_width(0.2)
        self.cell(130, 5, self.running_head)
        self.cell(0, 5, self.generated, align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(LEFT, 15, RIGHT, 15)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, f"{self.page_no()} / {{nb}}", align="C")

    def chip(self, text, fill):
        self.set_font("Helvetica", "B", 7)
        self.set_char_spacing(0.5)
        w = self.get_string_width(text) + 7
        self.set_fill_color(*fill)
        self.rect(LEFT, self.get_y(), w, 6, style="F", round_corners=True, corner_radius=3)
        self.set_text_color(*INK)
        self.cell(w, 6, text, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_char_spacing(0)

    def opener(self, chip, fill, text, lede, size=22):
        """Section opener: coloured chip, title, a short description."""
        self.chip(chip, fill)
        self.ln(3)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*INK)
        self.multi_cell(0, size * 0.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)
        self.paragraph(lede, 9, MUTED, 4.8)
        self.ln(2)
        self.line(LEFT, self.get_y(), RIGHT, self.get_y())
        self.ln(3)

    def subheading(self, text, note=None):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*INK)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        if note:
            self.paragraph(note, 8, MUTED, 4)
            self.ln(1)

    def paragraph(self, text, size=9, color=MUTED, line=5):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        self.multi_cell(0, line, text, new_x="LMARGIN", new_y="NEXT")

    def keep(self, folder, entry, extra):
        """Start a new page unless the figure and `extra` mm above it fit on this one."""
        image = Image.open(folder / entry["file"])
        if image.height / image.info.get("dpi", (300, 300))[0] * 25.4 + extra > 297 - BOTTOM - self.get_y():
            self.add_page()

    def image_at(self, folder, entry, x=None, y=None):
        """A figure at its own size (the figures are drawn for the page), centred unless x is given; returns its height."""
        image = Image.open(folder / entry["file"])
        dpi = image.info.get("dpi", (300, 300))[0]
        w, h = image.width / dpi * 25.4, image.height / dpi * 25.4
        room = 297 - BOTTOM - (self.get_y() if y is None else y)
        if h > room:
            self.add_page()
        y = self.get_y() if y is None else y
        self.image(str(folder / entry["file"]), x=(210 - w) / 2 if x is None else x, y=y, w=w)
        self.set_y(y + h + 3)
        return w, h


def drivers(depth):
    return [x for x in depth["drivers"] if x["st"] >= 0.01]


def hero(pdf, stats):
    unit, headline = stats["header"]["qoi_unit"], stats["headline"]
    peak, top = stats["depths"][headline]["peak"], next(iter(drivers(stats["depths"][headline])), None)
    items = [(sci(peak["median"]), f"MEDIAN PEAK, {headline.upper()}", unit), (sci(peak["high"]), "95TH PERCENTILE PEAK", unit),
             ("\n".join(top["label"].rsplit(" ", 1)) if top else "none", "MOST SENSITIVE PARAMETER", None)]
    y = pdf.get_y() + 3
    for i, (value, label, suffix) in enumerate(items):
        x = LEFT + i * 62
        pdf.set_text_color(*INK)
        if suffix:
            pdf.set_font("Helvetica", "B", 18)
            pdf.text(x, y + 9, value)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*MUTED)
            pdf.text(x, y + 20.5, suffix)
        else:
            pdf.set_xy(x, y + 1)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(56, 5.5, value)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.set_char_spacing(0.4)
        pdf.set_text_color(*MUTED)
        pdf.text(x, y + 16.5, label)
        pdf.set_char_spacing(0)
    pdf.set_y(y + 24)
    for label, d in stats["depths"].items():
        if "threshold" in d:
            t = d["threshold"]
            pdf.set_fill_color(*LEVEL[t["level"]])
            pdf.ellipse(LEFT, pdf.get_y() + 1.4, 2.6, 2.6, "F")
            pdf.set_x(LEFT + 6)
            when = "never exceeds" if t["p95_exceeds"] is None else f"exceeds from t = {sci(t['p95_exceeds'], 0)} yr"
            pdf.paragraph(f"{label}: exceeds {sci(t['value'], 1)} {unit} in {t['share'] * 100:.0f} % of samples; P95 {when}", 8.5, INK, 5)


def table(pdf, stats):
    pdf.paragraph(f"Peak over time, in {stats['header']['qoi_unit']}", 7.5, MUTED, 4)
    heads, widths = ["Location", "Median peak", "95th percentile", "Most sensitive parameter"], (34, 34, 34, 78)
    pdf.set_draw_color(*HAIRLINE)
    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "", 8.5)
    with pdf.table(borders_layout="HORIZONTAL_LINES", col_widths=widths, line_height=6.5, text_align=("LEFT",) * len(widths), cell_fill_color=(255, 255, 255),
                   cell_fill_mode="ALL", headings_style=FontFace(emphasis="BOLD", size_pt=7.5, color=MUTED, fill_color=(255, 255, 255))) as rows:
        rows.row(heads)
        for label, d in stats["depths"].items():
            top = next(iter(drivers(d)), None)
            rows.row([label, sci(d["peak"]["median"]), sci(d["peak"]["high"]), f"{top['label']}  (ST {top['st']:.2f})" if top else "-"])


def flatten(block, prefix=""):
    """(label, text) per leaf of a nested settings block; empty values are left out."""
    rows = []
    for key, value in block.items():
        label = f"{prefix} · {key}" if prefix else key
        if isinstance(value, dict):
            rows += flatten(value, label)
        elif value not in (None, [], [None, None], [[None, None]]):
            if isinstance(value, bool):
                text = "yes" if value else "no"
            elif isinstance(value, list) and len(value) > 6 and all(isinstance(v, (int, float)) for v in value):
                text = f"{len(value)} values, {min(value):g} to {max(value):g}"
            elif isinstance(value, list):
                text = ", ".join(str(v) for v in value)
            else:
                text = str(value)
            rows.append((label.replace("_", " ").capitalize(), text))
    return rows


def settings(pdf, x, y, width, groups):
    """Groups of (label, text) rows in one column starting at (x, y); returns the end y."""
    for name, rows in groups:
        pdf.set_xy(x, y + 2)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.set_char_spacing(0.4)
        pdf.set_text_color(*INK)
        pdf.cell(width, 4, name.upper())
        pdf.set_char_spacing(0)
        y = pdf.get_y() + 5
        pdf.set_draw_color(*HAIRLINE)
        pdf.line(x, y - 0.5, x + width, y - 0.5)
        for key, text in rows:
            pdf.set_font("Helvetica", "", 7.5)
            rows_needed = max(len(pdf.multi_cell(width * 0.5, 4.2, part, dry_run=True, output="LINES")) for part in (key, text))
            pdf.set_xy(x, y)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(width * 0.5, 4.2, key, align="L")
            pdf.set_xy(x + width * 0.5, y)
            pdf.set_text_color(*INK)
            pdf.multi_cell(width * 0.5, 4.2, text, align="L")
            y += 4.2 * rows_needed
    return y


def groups(block, general):
    """A block's nested settings as groups: one per sub-block, the plain entries under `general`; simulator internals and the options of other surrogate types are left out."""
    block = {k: v for k, v in block.items() if k not in SKIP and not (k.endswith("_opts") and k != f"{block.get('type')}_opts")}
    plain = {k: v for k, v in block.items() if not isinstance(v, dict)}
    return ([(general, flatten(plain))] if plain else []) + [(k.replace("_", " ").capitalize(), flatten(v)) for k, v in block.items() if isinstance(v, dict)]


def main():
    parser = argparse.ArgumentParser(description="PDF report of one scenario from its analysis folder.")
    parser.add_argument("--analysis", required=True, help="folder with stats.json, figures.json and the figures folder")
    parser.add_argument("--scenario", required=True, help="yaml of the scenario, for the settings")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = Path(args.analysis)
    figures = folder / "figures"
    stats, index = json.loads((folder / "stats.json").read_text()), json.loads((folder / "figures.json").read_text())
    head, scenario = stats["header"], read(Path(args.scenario))
    unit = head["qoi_unit"]
    n_inputs = sum(len(v) for group in (scenario.get("uncertain_parameters") or {}).values() for v in group.values())
    threshold = next((d["threshold"]["value"] for d in stats["depths"].values() if "threshold" in d), None)
    evolution = [e for e in index if e["plot"] == "profile"] or [e for e in index if e["plot"] == "timeseries"]
    depths = list(dict.fromkeys(e["depth"] for e in index if e["depth"] is not None))
    places = ", ".join(depths)

    pdf = Report(f"{head['scenario'].replace('_', ' ')}  ·  {head['engine_name']}")
    pdf.add_page()
    pdf.opener("SUMMARY", SUMMARY, stats["qoi"], f"{head['scenario'].replace('_', ' ')}: uncertainty and sensitivity of the {stats['qoi']} at {places}, predicted by the surrogate from {head['n_samples']:,} random samples of the uncertain inputs.")
    hero(pdf, stats)
    pdf.ln(2)
    table(pdf, stats)
    if evolution:
        pdf.ln(3)
        pdf.paragraph(f"The {stats['qoi']} " + ("over depth at the times shown" if evolution[0]["plot"] == "profile" else "over time at the locations shown")
                      + ": median (dashed), 5-95 % band" + (", threshold (dotted red)" if threshold else "") + ".", 8, MUTED, 4)
        for entry in evolution:
            if entry["depth"] is not None:
                pdf.keep(figures, entry, 6)
                pdf.paragraph(entry["depth"], 8.5, INK, 5)
            pdf.image_at(figures, entry)


    entries = [e for e in index if e["plot"] in ("stratigraphy", "inputs")]
    if entries:
        pdf.add_page()
        pdf.opener("INFORMATION", INFORMATION, "Inputs", "The stratigraphy and the " + (f"{n_inputs} " if n_inputs else "") + "uncertain inputs of the model, each a probability distribution per rock layer, propagated through the surrogate to the prediction.")
        x, tallest = LEFT, 0
        for entry in entries:
            w, h = Image.open(figures / entry["file"]).size
            dpi = Image.open(figures / entry["file"]).info.get("dpi", (300, 300))[0]
            pdf.image(str(figures / entry["file"]), x=x, y=pdf.get_y() + 1, w=w / dpi * 25.4)
            x, tallest = x + w / dpi * 25.4 + 4, max(tallest, h / dpi * 25.4)
        pdf.set_y(pdf.get_y() + tallest + 3)

    if scenario:
        pdf.add_page()
        surrogate = scenario.get("surrogate", {})
        n_train, n_valid = nested(surrogate, "sampling", "training", "sample_size"), nested(surrogate, "sampling", "validation", "sample_size")
        pdf.opener("PREDICTION", PREDICTION, "Model and surrogate", "A fast surrogate model stands in for the simulator" + (f". It is trained on {n_train} simulations" if n_train else "")
                  + (f" and checked against {n_valid} independent ones" if n_valid else "") + ".")
        if stats.get("validation"):
            pdf.subheading("Validation", "R² of the surrogate against the simulator at each location, over the output times; 1 is a perfect reproduction.")
            with pdf.table(borders_layout="HORIZONTAL_LINES", col_widths=(60, 40, 40), line_height=6.5, text_align=("LEFT",) * 3, cell_fill_color=(255, 255, 255), cell_fill_mode="ALL",
                           headings_style=FontFace(emphasis="BOLD", size_pt=7.5, color=MUTED, fill_color=(255, 255, 255))) as rows:
                pdf.set_font("Helvetica", "", 8.5)
                rows.row(["Location", "R² minimum", "R² mean"])
                for name, v in stats["validation"].items():
                    rows.row([name, f"{v['r2_min']:.3f}", f"{v['r2_mean']:.3f}"])
            pdf.ln(2)
        pdf.subheading("Settings")
        y = pdf.get_y()
        left = settings(pdf, LEFT, y, 85, [("Site", flatten({"site": nested(scenario, "communication_sdh", "site_name"), "nuclides": nested(scenario, "communication_ntd", "nuclide_to_consider")}))] + groups(scenario.get("model", {}), "Model"))
        right = settings(pdf, LEFT + 95, y, 85, groups(surrogate, "Surrogate"))
        pdf.set_y(max(left, right) + 4)

    for chip, fill, title, plots in SECTIONS:
        entries = [e for e in index if e["plot"] in plots and e not in evolution]
        if not entries:
            continue
        pdf.add_page()
        if "dependence" in plots:
            lede = (f"Total-order Sobol index ST of each uncertain input: the share of the variance of the {stats['qoi']} it causes, alone and through interactions "
                    "(0 none, 1 all). Heatmaps show it over time, bars at the time of the peak, scatter plots the sampled input-output relation with its correlation r.")
        else:
            lede = (f"The input uncertainty propagated through the surrogate with {head['n_samples']:,} random samples: median, 5-95 % band" + (f", and the threshold of {sci(threshold, 1)} {unit}" if threshold else "")
                    + ". Histograms show the distribution at one time.")
        pdf.opener(chip, fill, title, lede)
        order = {plot: n for n, plot in enumerate(plots)}
        for k, depth in enumerate(dict.fromkeys(e["depth"] for e in entries)):
            if k and "dependence" in plots:  # sensitivity: one subsection per location, each on its own page
                pdf.add_page()
            for n, entry in enumerate(sorted((e for e in entries if e["depth"] == depth), key=lambda e: order[e["plot"]])):
                if entry["plot"] == "dependence":  # the input-output scatter grids get their own page
                    pdf.add_page()
                    pdf.subheading(depth, "Sampled input-output relation: each point is one sample, r is the correlation of the input with the output.")
                elif n == 0:
                    pdf.keep(figures, entry, 12)
                    pdf.subheading(depth)
                pdf.image_at(figures, entry)

    pdf.add_page()
    pdf.opener("RUN INFORMATION", SUMMARY, "Run information", "What produced this report, for reproducing and tracing it.")
    sampling, provenance = surrogate.get("sampling", {}) if scenario else {}, stats.get("provenance", {})
    identity = [("Scenario", head["scenario"].replace("_", " ")), ("Scenario hash", provenance.get("scenario_hash")), ("UQ hash", provenance.get("uq_hash")), ("Analysis hash", provenance.get("analysis_hash")), ("Engine", f"{head['engine_name']} ({head['surrogate_type']})"), ("Output", f"{head['output_qoi']} [{unit}]")] + ([("Conversion", f"{provenance['raw_unit']} × {provenance['qoi_factor']:g}")] if provenance.get("qoi_factor", 1) != 1 else [])
    samples = [(f"{name.capitalize()} samples", f"{v['sample_size']}, {v['sampling_method']}, seed {v['seed']}") for name, v in sampling.items() if isinstance(v, dict) and "sample_size" in v]
    y = pdf.get_y()
    left = settings(pdf, LEFT, y, 85, [("Identity", [row for row in identity if row[1]]), ("Dates", [("Engine computed", head["created"][:16].replace("T", " ")), ("Report generated", pdf.generated)])])
    right = settings(pdf, LEFT + 95, y, 85, [("Sampling", samples + [("Propagation samples", f"{head['n_samples']:,}, random, seed {head['seed']}")]), ("Input data (content hash)", list(provenance.get("data", {}).items()))])
    pdf.output(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
