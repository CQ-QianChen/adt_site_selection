import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import NullFormatter
from scipy.stats import gaussian_kde

from . import layer_colors, setup, signed_threshold
from ..io import depth_index, time_label


def profile(propagation, layers, path, times=None, z_min=None, z_max=None, stratigraphy=True, boundaries=True, marks=None, inset_depths=None, inset_bounds=None,
            threshold=None, legend=True, title=True, text=None, style=None):
    """Median and percentile band against depth at the nearest `times`; `stratigraphy` adds the layer column, `boundaries` the layer lines (named by `marks`, depth -> name), `threshold` a vertical line, `inset_depths` (one time) distribution insets."""
    s, t, k, rc = setup(propagation, style, text, 1)
    output, time, z = propagation["output/mean"], propagation["axes/time"], propagation["axes/depth_z"]
    stratigraphy = stratigraphy and layers is not None
    slots = (inset_bounds or s["profile_insets"])[:len(inset_depths or [])]
    idx = np.arange(len(time)) if times is None else [np.abs(time - x).argmin() for x in times]
    if len(idx) > s["max_panels"]:
        raise ValueError(f"profile shows at most {s['max_panels']} times (style max_panels), pass times")
    z_min, z_max = z.min() if z_min is None else z_min, z.max() if z_max is None else z_max
    keep = (z >= z_min) & (z <= z_max)
    z = z[keep]
    shown = [(name, min(top, z_max), max(bottom, z_min)) for name, top, bottom in layers if top > z_min and bottom < z_max] if layers is not None else []
    low, median, high = np.percentile(output[:, idx][:, :, keep], [s["band"][0], 50, s["band"][1]], axis=0)
    label = f"{t['quantity']} [{t['unit']}]"
    limit = signed_threshold(threshold, np.concatenate([low.ravel(), high.ravel()]))
    pt = s["font"] / 72  # inches per font point: margins for tick labels, axis labels and titles
    m_left, m_right, m_bottom, m_top = 4.2 * pt, 0.5 * pt, (3.5 + 1.3 * (len(idx) > 1)) * pt, (3 + 0.8 * bool(t["qoi"])) * pt  # a second line of x label for several panels, a line for the QoI text
    n_axes = len(idx) + stratigraphy
    overhang = 2 * max([b[0] + b[2] - 1 for b in slots] + [0])  # of the insets, in panel units (a panel is 2 wide, the column 1.5)
    units = (1.5 * stratigraphy + 2 * len(idx)) * (1 + 0.3 * (n_axes - 1) / n_axes) + overhang
    u = min((s["size"][0] - m_left - m_right) / units, 0.095 * s["font"])  # a lone panel stays as narrow as in a multi-panel figure
    if not s["height"]:
        s["size"] = (m_left + m_right + u * units, m_bottom + m_top + 2 * s["profile_aspect"] * u)
    fig_w, fig_h = s["size"]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(idx) + stratigraphy, sharey=True, squeeze=False, figsize=s["size"],
                                 gridspec_kw={"width_ratios": [1.5] * stratigraphy + [2] * len(idx)})
        axes = axes[0]
        if stratigraphy:
            for (name, top, bottom), c in zip(shown, layer_colors(s, len(shown))):
                axes[0].axhspan(bottom, top, color=c)
                axes[0].axhline(bottom, color="0.5", lw=0.8 * k)
                axes[0].text(0.5, (top + bottom) / 2, textwrap.fill(name.replace("_", " "), s["wrap"], break_long_words=False), ha="center", va="center",
                             fontsize=0.7 * s["font"], transform=axes[0].get_yaxis_transform())
            axes[0].set(xticks=[], title=t["stratigraphy"] if title else "")
            axes[0].set_box_aspect(s["profile_aspect"] * 2 / 1.5)
        axes[0].set_ylabel(t["depth"])
        for ax, i, x in zip(axes[stratigraphy:], range(len(idx)), time[idx]):
            ax.fill_betweenx(z, low[i], high[i], color=s["color"], alpha=0.25, label=t["band"].format(low=s["band"][0], high=s["band"][1]))
            ax.plot(median[i], z, "--", color=s["color"], lw=s["median_lw"] * k, label=t["median"])
            for bottom in [b for _, _, b in shown if b > z_min] if boundaries else []:
                ax.axhline(bottom, color="0.7", lw=1.2 * k, ls="--", zorder=0)
            for d, name in (marks or {}).items():
                ax.text(0.98, d, name, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=0.7 * s["font"], color="0.4")
            if limit is not None:
                ax.axvline(limit, color=s["threshold_color"], lw=2 * k, ls=":", label=t["threshold"])
            ax.set(xlim=(0, max(high.max(), limit or 0) * 1.1), title=time_label(x) if title else "", xlabel=label if len(idx) == 1 else label.replace(" [", "\n["))
            ax.locator_params(axis="x", nbins=2 if len(idx) > 1 or inset_depths else 3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_box_aspect(s["profile_aspect"])
        axes[0].set_ylim(z_min, z_max)
        ax = axes[-1]
        if legend:
            ax.legend(loc="best" if legend is True else legend)
        if inset_depths:
            if len(idx) != 1:
                raise ValueError("inset_depths needs exactly one time")
            if len(slots) < len(inset_depths):
                raise ValueError("inset_bounds needs one entry per depth")
            for d, bounds in zip(sorted(inset_depths, reverse=True), slots):
                j = np.abs(z - d).argmin()
                inset = ax.inset_axes(bounds)
                distribution(propagation, None, z[j], time[idx[0]], ticks=False, legend=False, title=False, text=text,
                             style={**(style or {}), "linewidth": 0.45}, ax=inset)
                inset.set_xlabel("")
                ax.plot([low[0][j], high[0][j]], [z[j], z[j]], color=s["color"], lw=1.5 * k, solid_capstyle="butt", clip_on=False)
                ax.plot(median[0][j], z[j], "o", color=s["color"], ms=7 * k, clip_on=False)
                x0, x1 = inset.get_xlim()
                link = ConnectionPatch((median[0][j], z[j]), ((median[0][j] - x0) / (x1 - x0), 0 if bounds[1] > (z[j] - z_min) / (z_max - z_min) else 1),
                                       coordsA="data", coordsB="axes fraction", axesA=ax, axesB=inset, color=s["color"], lw=s["median_lw"] * k, ls=":")
                ax.add_artist(link)
                link.set_clip_on(False)
        fig.subplots_adjust(left=m_left / fig_w, right=1 - (overhang * u + m_right) / fig_w, bottom=m_bottom / fig_h, top=1 - m_top / fig_h, wspace=0.3)
        if t["qoi"]:
            fig.text(0.5, 0.99, t["qoi"], ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
        fig.savefig(path, dpi=300)
        plt.close(fig)


def timeseries(propagation, layers, path, depth, t_min=None, t_max=None, log_x=False, log_y=False, stratigraphy=False, z_min=None, z_max=None,
               inset_time=None, inset_bounds=None, threshold=None, legend=True, title=True, text=None, style=None):
    """Median and percentile band over time at one depth; `stratigraphy` adds the layer column, `threshold` a horizontal line, `inset_time` a distribution inset."""
    s, t, k, rc = setup(propagation, style, text, 0.58)
    output, time, z = propagation["output/mean"], propagation["axes/time"], propagation["axes/depth_z"]
    i, where = depth_index(propagation, depth)
    where = t["where"] or where
    t_min, t_max = time[0] if t_min is None else t_min, time[-1] if t_max is None else t_max
    keep = (time >= t_min) & (time <= t_max) & (time > 0 if log_x else True)
    time = time[keep]
    low, median, high = np.percentile(output[:, keep, i], [s["band"][0], 50, s["band"][1]], axis=0)
    finite = z[np.isfinite(z)]
    pad = max(np.ptp(finite) / 2, 100) if len(finite) else 0
    lo = (finite.min() - pad if len(finite) else layers[-1][2]) if z_min is None else z_min
    hi = (finite.max() + pad if len(finite) else layers[0][1]) if z_max is None else z_max
    stratigraphy = stratigraphy and layers is not None

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 1 + stratigraphy, squeeze=False, figsize=s["size"], gridspec_kw={"width_ratios": [1.5] * stratigraphy + [4]})
        axes = axes[0]
        ax = axes[-1]
        if stratigraphy:
            shown = [(name, min(top, hi), max(bottom, lo)) for name, top, bottom in layers if top > lo and bottom < hi]
            for (name, top, bottom), c in zip(shown, layer_colors(s, len(shown))):
                axes[0].axhspan(bottom, top, color=c)
                axes[0].axhline(bottom, color="0.5", lw=0.8 * k)
                axes[0].text(0.5, (top + bottom) / 2, textwrap.fill(name.replace("_", " "), s["wrap"], break_long_words=False), ha="center", va="center",
                             fontsize=0.7 * s["font"], transform=axes[0].get_yaxis_transform())
            if np.isfinite(z[i]):
                axes[0].axhline(z[i], color=s["color"], lw=5 * k)
            axes[0].set(xticks=[], title=t["stratigraphy"] if title else "", ylabel=t["depth"], ylim=(lo, hi))
        ax.fill_between(time, low, high, color=s["color"], alpha=0.25, label=t["band"].format(low=s["band"][0], high=s["band"][1]))
        ax.plot(time, median, "--", color=s["color"], lw=s["median_lw"] * k, label=t["median"])
        limit = signed_threshold(threshold, np.concatenate([low, high]))
        if limit is not None:
            ax.axhline(limit, color=s["threshold_color"], lw=2 * k, ls=":", label=t["threshold"])
        ax.set(xlabel=t["time"], ylabel=f"{t['quantity']} [{t['unit']}]", title=t["title"].format(where=where, time="").strip() if title else "", xlim=(time[0] if log_x else t_min, t_max),
               ylim=(0 if low.min() >= 0 else None, None), xscale="log" if log_x else "linear", yscale="log" if log_y else "linear")
        ax.grid(alpha=0.3, lw=0.8 * k)
        ax.spines[["top", "right"]].set_visible(False)
        if log_x:
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.locator_params(axis="x", nbins=4)
            ax.ticklabel_format(axis="x", style="plain")
        if legend:
            ax.legend(loc="best" if legend is True else legend)
        if inset_time is not None:
            n = np.abs(time - inset_time).argmin()
            bounds = inset_bounds or s["timeseries_inset"]
            inset = ax.inset_axes(bounds)
            distribution(propagation, None, depth, inset_time, ticks=False, legend=False, title=False, text=text, style={**(style or {}), "linewidth": 0.45},
                         ax=inset)
            inset.set_xlabel("")
            ax.plot([time[n], time[n]], [low[n], high[n]], color=s["color"], lw=1.5 * k, solid_capstyle="butt", clip_on=False)
            ax.plot(time[n], median[n], "o", color=s["color"], ms=7 * k, clip_on=False)
            py = ax.transAxes.inverted().transform(ax.transData.transform((time[n], median[n])))[1]
            x0, x1 = inset.get_xlim()
            link = ConnectionPatch((time[n], median[n]), ((median[n] - x0) / (x1 - x0), 0 if bounds[1] > py else 1), coordsA="data", coordsB="axes fraction",
                                   axesA=ax, axesB=inset, color=s["color"], lw=s["median_lw"] * k, ls=":")
            ax.add_artist(link)
            link.set_clip_on(False)
        if t["qoi"]:
            fig.text(0.5, 0.99, t["qoi"], ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, dpi=300)
        plt.close(fig)


def distribution(propagation, path, depth, time, threshold=None, ticks=True, legend=True, title=True, text=None, style=None, ax=None):
    """Histogram and KDE of the samples at one depth and time, `threshold` a vertical line; with `ax` it draws into that axes."""
    s, t, k, rc = setup(propagation, style, text, 0.58)
    i, where = depth_index(propagation, depth)
    where = t["where"] or where
    n = np.abs(propagation["axes/time"] - time).argmin()
    values = propagation["output/mean"][:, n, i]
    edges = np.linspace(values.min(), values.max(), s["bins"] + 1)
    counts, _ = np.histogram(values, bins=edges, density=True)
    grid = np.linspace(values.min(), values.max(), 400)
    low, median, high = np.percentile(values, [s["band"][0], 50, s["band"][1]])
    median_lw = s["median_lw"] * k  # the same as the median of the plot the inset belongs to
    k *= s["linewidth"]

    with plt.rc_context(rc):
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=s["size"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=s["bar_color"], edgecolor="white", linewidth=k)
        ax.fill_between(grid, gaussian_kde(values)(grid), color=s["kde_color"], alpha=0.7, lw=0)
        ax.axvline(low, color=s["band_color"], lw=2 * k, label=t["band"].format(low=s["band"][0], high=s["band"][1]))
        ax.axvline(high, color=s["band_color"], lw=2 * k)
        ax.axvline(median, color=s["color"], lw=median_lw, ls="--", label=t["median"])
        limit = signed_threshold(threshold, values)
        if limit is not None:
            ax.axvline(limit, color=s["threshold_color"], lw=2 * k, ls=":", label=t["threshold"])
        ax.set(xlabel=f"{t['quantity']} [{t['unit']}]", yticks=[], title=t["title"].format(where=where, time=f" [{time_label(propagation['axes/time'][n])}]").strip() if title else "", ylim=(0, None))
        if not ticks:
            ax.set(xticklabels=[], yticklabels=[])
            ax.tick_params(which="both", length=0)
        if legend:
            ax.legend(loc="best" if legend is True else legend)
        if own:
            if t["qoi"]:
                fig.text(0.5, 0.99, t["qoi"], ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(path, dpi=300)
            plt.close(fig)


def dependence(propagation, path, depth, time=None, inputs=None, rows=None, threshold=None, title=True, text=None, style=None):
    """Output against inputs at one depth and time (default: the last), as density maps in a grid; `inputs` is all of `rows` (default), a count (the most correlated) or a list of labels, `rows` are (label, index); `threshold` is a horizontal line."""
    s, t, k, rc = setup(propagation, style, text, 1)
    i, where = depth_index(propagation, depth)
    where = t["where"] or where
    times = propagation["axes/time"]
    n = len(times) - 1 if time is None else np.abs(times - time).argmin()
    X, y = propagation["input/X"], propagation["output/mean"][:, n, i]
    rows = rows or list(zip(propagation["axes/parameter_names"], range(X.shape[1])))
    r = np.nan_to_num([np.corrcoef(X[:, j], y)[0, 1] for _, j in rows])
    limit = signed_threshold(threshold, y)
    top = list(range(len(rows))) if inputs is None else np.argsort(-np.abs(r))[:inputs] if isinstance(inputs, int) else [[label for label, _ in rows].index(name) for name in inputs]
    pt = s["font"] / 72  # inches per font point
    lines = max(len(textwrap.wrap(rows[m][0], s["wrap"], break_long_words=False)) for m in top)  # lines of the longest input label
    cols = min(len(top), s["dependence_columns"])
    n_rows = -(-len(top) // cols)
    m_left, m_right, m_top = 4.2 * pt, 0.5 * pt, (1.2 + 1.2 * bool(title) + 1.2 * bool(t["qoi"])) * pt
    label = (3.4 + 1.2 * (lines + bool(title))) * pt  # tick labels and x label under each panel
    pw = (s["size"][0] - m_left - m_right) / (cols + 0.35 * (cols - 1))
    if not s["height"]:
        s["size"] = (s["size"][0], m_top + n_rows * (label + s["dependence_aspect"] * pw))
    fig_w, fig_h = s["size"]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, cols, sharey=True, squeeze=False, figsize=s["size"])
        for ax in axes.flat[len(top):]:
            ax.axis("off")
        for ax, m in zip(axes.flat, top):
            ax.hexbin(X[:, rows[m][1]], y, gridsize=s["gridsize"], cmap=s["hexbin_cmap"], bins="log", mincnt=1, linewidths=0)
            if limit is not None:
                ax.axhline(limit, color=s["threshold_color"], lw=2 * k, ls=":")
            ax.set_xlabel(textwrap.fill(rows[m][0], s["wrap"], break_long_words=False) + (f"\nr = {r[m]:+.2f}" if title else ""), labelpad=s["font"] * 0.9)
            ax.locator_params(axis="x", nbins=3)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{t['quantity']} [{t['unit']}]")
        fig.subplots_adjust(left=m_left / fig_w, right=1 - m_right / fig_w, bottom=label / fig_h, top=1 - m_top / fig_h, wspace=0.35, hspace=label / (s["dependence_aspect"] * pw))
        if t["qoi"]:
            fig.text(0.5, 1 - 0.1 * pt / fig_h, t["qoi"], ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
        if title:
            fig.text(0.5, 1 - (0.1 + 1.2 * bool(t["qoi"])) * pt / fig_h, t["title"].format(where=where, time=f" [{time_label(times[n])}]").strip(), ha="center", va="top", fontsize=plt.rcParams["axes.titlesize"])
        fig.savefig(path, dpi=300)
        plt.close(fig)

