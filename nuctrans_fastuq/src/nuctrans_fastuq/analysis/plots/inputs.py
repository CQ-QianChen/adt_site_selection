import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from scipy.stats import gaussian_kde

from . import layer_colors, setup
from ..io import parameter_rows


def stratigraphy(propagation, layers, path, z_min=None, z_max=None, title=True, text=None, style=None):
    """The layers as coloured blocks against depth."""
    s, t, k, rc = setup(propagation, style, text, 1.6)
    z_min, z_max = layers[-1][2] if z_min is None else z_min, layers[0][1] if z_max is None else z_max
    shown = [(name, min(top, z_max), max(bottom, z_min)) for name, top, bottom in layers if top > z_min and bottom < z_max]

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=s["size"])
        for (name, top, bottom), c in zip(shown, layer_colors(s, len(shown))):
            ax.axhspan(bottom, top, color=c)
            ax.axhline(bottom, color="0.5", lw=0.8 * k)
            ax.text(0.5, (top + bottom) / 2, textwrap.fill(name.replace("_", " "), s["wrap"], break_long_words=False), ha="center", va="center",
                    fontsize=0.7 * s["font"], transform=ax.get_yaxis_transform())
        ax.set(xticks=[], ylim=(z_min, z_max), ylabel=t["depth"], title=t["stratigraphy"] if title else "")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


def inputs(propagation, layers, path, rows=None, units=None, title=True, text=None, style=None):
    """Distribution of the sampled inputs as ridgelines, one panel per property (the last word of a label) with a ridge per layer, coloured from top to bottom; `rows` are (label, index), `units` maps a label to its unit."""
    s, t, k, rc = setup(propagation, style, text, 1)
    X = propagation["input/X"]
    rows = rows or parameter_rows(propagation["axes/parameter_names"], layers)
    groups = {}
    for label, j in rows:
        name, _, prop = label.rpartition(" ")
        groups.setdefault(prop, []).append((name, j, label))
    pt = s["font"] / 72  # inches per font point
    heights = [(0.77 * (len(g) - 1) + 1.2) * 3.8 * pt for g in groups.values()]  # ridges overlap by 30% of their height
    gap, m_left, m_top, m_bottom = 5 * pt, 9 * pt, 1.8 * pt, 3.2 * pt
    if not s["height"]:
        s["size"] = (s["size"][0], m_top + m_bottom + sum(heights) + gap * len(groups))
    fig_w, fig_h = s["size"]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(groups), 1, squeeze=False, figsize=s["size"], gridspec_kw={"height_ratios": heights})
        for ax, (prop, group) in zip(axes[:, 0], groups.items()):
            values = [X[:, j] for _, j, _ in group]
            low, high = min(v.min() for v in values), max(v.max() for v in values)
            grid = np.linspace(low - 0.05 * (high - low), high + 0.05 * (high - low), 400)
            colors = colormaps[s["ridge_cmap"]](np.linspace(*s["ridge_range"], len(group)))
            for n, ((name, _, _), v) in enumerate(zip(group, values)):
                density = gaussian_kde(v)(grid)
                density = density / density.max()
                base = (len(group) - 1 - n) * 0.77
                ax.fill_between(grid, base, base + density, color=colors[n], alpha=0.75, lw=0)
                ax.plot(grid, base + density, color="black", lw=k)
            ax.set(yticks=[(len(group) - 1 - n) * 0.77 for n in range(len(group))], yticklabels=[textwrap.fill(name.replace("/", "/ "), s["wrap"], break_long_words=False) for name, _, _ in group],
                   ylim=(0, None))
            if title:
                unit = (units or {}).get(group[0][2])
                ax.set_title(prop.capitalize() + (f" [{unit}]" if unit else ""), loc="left", fontweight="bold")
            ax.tick_params(axis="y", length=0)
            ax.locator_params(axis="x", nbins=3)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
            ax.spines[["top", "right", "left"]].set_visible(False)
        fig.subplots_adjust(left=m_left / fig_w, right=1 - 0.5 * pt / fig_w, bottom=m_bottom / fig_h, top=1 - m_top / fig_h, hspace=gap / (sum(heights) / len(groups)))
        fig.savefig(path, dpi=300)
        plt.close(fig)
