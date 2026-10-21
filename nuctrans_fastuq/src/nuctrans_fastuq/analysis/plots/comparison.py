import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from . import setup
from ..io import depth_index


def curves(runs, s):
    """Percentiles of the magnitude of the output at one depth per run, the lowest value and the first time that the plots show."""
    out = []
    for label, propagation, depth, *_ in runs:
        i = depth_index(propagation, depth)[0]
        out.append((label, propagation["axes/time"], np.percentile(np.abs(propagation["output/mean"][:, :, i]), [s["band"][0], 50, s["band"][1]], axis=0)))
    floor = min(curve[2].max() for *_, curve in out) * s["floor"]
    return out, floor, min(time[curve[2] >= floor][0] for _, time, curve in out)


def timeseries(runs, path, threshold=None, text=None, style=None):
    """Median and percentile band over time of several runs on shared log axes, `runs` a list of (label, propagation, depth); magnitudes, so that outputs of either sign compare."""
    s, t, k, rc = setup(runs[0][1], style, text, 0.58)
    curve, floor, start = curves(runs, s)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=s["size"])
        for (label, time, (low, median, high)), color in zip(curve, s["run_colors"]):
            ax.fill_between(time[time > 0], low[time > 0], high[time > 0], color=color, alpha=0.2, lw=0)
            ax.plot(time[time > 0], median[time > 0], color=color, lw=s["median_lw"] * k, label=label)
        if threshold is not None:
            ax.axhline(threshold, color=s["threshold_color"], lw=2 * k, ls=":", label=t["threshold"])
        ax.set(xlabel=t["time"], ylabel=f"|{t['quantity']}| [{t['unit']}]", xscale="log", yscale="log", xlim=(start, None), ylim=(floor, None))
        ax.grid(alpha=0.3, lw=0.8 * k)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


def site_map(runs, regions, path, threshold=None, text=None, style=None):
    """A map of the regions with one marker per run, linked to its time series (median and band of the magnitude, north to south, shared axes); `runs` a list of (label, propagation, depth, region), `regions` {name: [(lon, lat) rings]}."""
    s, t, k, rc = setup(runs[0][1], style, text, 0.62)

    def project(lon, lat, lon0=10.45, lat0=51.16, p1=48.67, p2=54.0):  # Lambert conformal conic
        lon, lat, lon0, lat0, p1, p2 = (np.radians(v) for v in (lon, lat, lon0, lat0, p1, p2))
        n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
        f = np.cos(p1) * np.tan(np.pi / 4 + p1 / 2) ** n / n
        rho, rho0 = f / np.tan(np.pi / 4 + lat / 2) ** n, f / np.tan(np.pi / 4 + lat0 / 2) ** n
        return rho * np.sin(n * (lon - lon0)), rho0 - rho * np.cos(n * (lon - lon0))

    curve, floor, start = curves(runs, s)
    sites = []
    for (label, time, band), (_, _, _, region), color in zip(curve, runs, s["run_colors"]):
        x, y = max(regions[region], key=len).T  # area centroid of the main ring
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        marker = project(((x + np.roll(x, -1)) * cross).sum() / (3 * cross.sum()), ((y + np.roll(y, -1)) * cross).sum() / (3 * cross.sum()))
        sites.append({"label": label, "region": region, "color": color, "marker": marker, "time": time, "band": band})
    sites.sort(key=lambda site: -site["marker"][1])

    with plt.rc_context(rc):
        fig = plt.figure(figsize=s["size"])
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
            panel.plot(time[time > 0], median[time > 0], color=site["color"], lw=s["median_lw"] * k)
            if threshold is not None:
                panel.axhline(threshold, color=s["threshold_color"], lw=1.5 * k, ls=":")
            panel.set(xscale="log", yscale="log", xlim=(start, None), ylim=(floor, None))
            panel.set_title(f"{chr(ord('a') + n)}   {site['label']}", loc="left", fontweight="bold")
            panel.grid(alpha=0.3, lw=0.8 * k)
            panel.spines[["top", "right"]].set_visible(False)
            panel.xaxis.set_minor_formatter(NullFormatter())
            panel.tick_params(labelbottom=n == len(sites) - 1)
            panels.append(panel)
        panels[-1].set_xlabel(t["time"])
        fig.text(0.99, 0.995, f"|{t['quantity']}| [{t['unit']}]", ha="right", va="top")
        fig.canvas.draw()  # the map is final only after a draw
        for n, (site, panel) in enumerate(zip(sites, panels)):
            mx, my = site["marker"]
            ax.plot(mx, my, "o", color=site["color"], ms=7 * k, mec="white", mew=1.2 * k, zorder=5)
            ax.annotate(chr(ord("a") + n), (mx, my), xytext=(0, 9 * k), textcoords="offset points", ha="center", fontweight="bold", color=site["color"])
            a, b = fig.transFigure.inverted().transform(ax.transData.transform((mx, my))), panel.get_position()
            fig.add_artist(Line2D([a[0], b.x0 - 0.075], [a[1], (b.y0 + b.y1) / 2], transform=fig.transFigure, color=site["color"], lw=0.8 * k, alpha=0.7))
        fig.savefig(path, dpi=300)
        plt.close(fig)
