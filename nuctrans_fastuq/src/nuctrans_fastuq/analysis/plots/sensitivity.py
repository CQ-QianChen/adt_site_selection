import textwrap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullFormatter

from . import setup
from ..io import depth_index, time_label


def sobol_heatmap(sensitivity, path, depth, rows=None, index="ST", log_time=False, log_color=False, grid=True, equal_columns=None, title=True, text=None,
                  style=None):
    """Sobol index of every parameter over time at one depth; `rows` are (label, index) in display order."""
    s, t, k, rc = setup(sensitivity, style, text, 0.75)
    i, where = depth_index(sensitivity, depth)
    where = t["where"] or where
    rows = rows or list(zip(sensitivity["axes/parameter_names"], range(len(sensitivity["axes/parameter_names"]))))
    labels, order = [label for label, _ in rows], [j for _, j in rows]
    time = sensitivity["axes/time"]
    keep = time > 0 if log_time else slice(None)
    time, values = time[keep], sensitivity[index][order][:, keep, i]
    few = len(time) <= 6 if equal_columns is None else equal_columns
    if few:
        edges = np.arange(len(time) + 1) - 0.5
    else:
        axis = np.log10(time) if log_time else time
        middle = (axis[1:] + axis[:-1]) / 2
        edges = np.r_[2 * axis[0] - middle[0], middle, 2 * axis[-1] - middle[-1]]
        if log_time:
            edges = 10 ** edges
    if log_color:
        values, norm = np.clip(values, s["floor"], 1), LogNorm(s["floor"], 1)
    else:
        norm = Normalize(0, 1)

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=s["size"])
        mesh = ax.pcolormesh(edges, np.arange(len(labels) + 1) - 0.5, values, cmap=plt.get_cmap(s["sobol_cmap"]).with_extremes(bad="white"), norm=norm,
                             edgecolors="0.75" if grid and few else "none", linewidth=0.8 * k)
        ax.set(xlabel=t["time"], title=t["title"].format(where=where, time="").strip() if title else "", yticks=range(len(labels)), yticklabels=[textwrap.fill(label, 2.5 * s["wrap"], break_long_words=False) for label in labels])
        ax.invert_yaxis()
        if few:
            ax.set(xticks=range(len(time)), xticklabels=[f"{x:.0e}".replace("e+0", "e") for x in time])
        else:
            ax.set(xscale="log" if log_time else "linear", xlim=(time[0], time[-1]))
            if log_time:
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.ticklabel_format(axis="x", style="plain")
        if grid and not few:
            ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
            ax.tick_params(axis="y", which="minor", length=0)
            ax.grid(axis="x", which="major", color="0.75", linewidth=0.8 * k)
            ax.grid(axis="x", which="minor", color="0.88", linewidth=0.6 * k)
            ax.grid(axis="y", which="minor", color="0.75", linewidth=0.8 * k)
        fig.colorbar(mesh, label=t["sobol"].format(index=index))
        if t["qoi"]:
            fig.text(0.5, 0.99, t["qoi"], ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94 if title or t["qoi"] else 1))
        fig.savefig(path, dpi=300)
        plt.close(fig)


def sobol_ranking(sensitivity, path, depth, rows=None, time=None, index="ST", min_sensitivity=0.01, title=True, text=None, style=None):
    """Parameters ranked by Sobol index at one depth and time; `rows` are (label, index)."""
    s, t, k, rc = setup(sensitivity, style, text, 0.75)
    i, where = depth_index(sensitivity, depth)
    where = t["where"] or where
    times = sensitivity["axes/time"]
    n = len(times) - 1 if time is None else np.abs(times - time).argmin()
    rows = rows or list(zip(sensitivity["axes/parameter_names"], range(len(sensitivity["axes/parameter_names"]))))
    values = np.array([sensitivity[index][j, n, i] for _, j in rows])
    labels = np.array([label for label, _ in rows])
    keep = np.flatnonzero(values >= min_sensitivity)
    keep = keep[np.argsort(values[keep])[::-1]]

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(s["size"][0], s["size"][0] * (3 + 0.8 * len(keep)) / 12) if not s["height"] else s["size"])
        ax.barh([textwrap.fill(label, 2.5 * s["wrap"], break_long_words=False) for label in labels[keep]], values[keep], color=plt.get_cmap(s["sobol_cmap"])(0.8))
        ax.invert_yaxis()
        ax.set(xlabel=t["sobol"].format(index=index), title=t["title"].format(where=where, time=f" [{time_label(times[n])}]").strip() if title else "", xlim=(0, 1))
        if t["qoi"]:
            fig.text(0.5, 0.99, t["qoi"], ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=300)
        plt.close(fig)
