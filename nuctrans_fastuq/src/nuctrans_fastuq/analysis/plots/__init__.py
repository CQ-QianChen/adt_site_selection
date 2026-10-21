import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

BASE_FONT = 22  # the layouts and line widths in the plots were tuned for this font size at REFERENCE_WIDTH inches wide and scale with the font
REFERENCE_WIDTH = 12

# defaults of every plot; the caller overrides any key through `style`
STYLE = {"width": 160, "height": None, "font": None, "min_font": 10, "linewidth": 1, "median_lw": 3, "band": (5, 95), "bins": 20, "floor": 1e-3, "wrap": 12, "max_panels": 4,
         "color": "#6B3580", "threshold_color": "#B94A4A", "bar_color": "#B0B0B0", "kde_color": "#E2D0EA", "band_color": "#C9A8D6", "layer_colors": ["#bdc2b0", "#b0b8ae", "#a9b6c0", "#9babb6"], "sobol_cmap": "YlGnBu", "ridge_cmap": "magma", "ridge_range": (0.15, 0.85),
         "profile_aspect": 2.93, "gridsize": 30, "hexbin_cmap": "Purples", "dependence_aspect": 1, "dependence_columns": 3, "timeseries_inset": (0.2, 0.56, 0.36, 0.38), "profile_insets": [(0.28, 0.58, 1.29, 0.294), (0.28, 0.015, 1.29, 0.294)]}
# words in the figures (where: replaces the depth in the title; title: template of the figure titles from where and the time); quantity, unit and qoi default to the names in the file, the caller overrides any key through `text` (qoi False: no top right text)
TEXT = {"depth": "Depth [m]", "time": "Time [yr]", "stratigraphy": "Stratigraphy", "sobol": "Sobol {index} index", "where": None, "title": "{where}{time}", "band": "{low}-{high}%", "median": "median", "threshold": "threshold"}


def setup(data, style, text, aspect):
    """Merge the defaults with `style` and `text`; the width gives the font (at least min_font) and the line widths follow it."""
    s = {**STYLE, **(style or {})}
    t = {"quantity": data["output_qoi"], "unit": data["qoi_unit"], "qoi": f"QoI: {data['output_qoi']}", **TEXT, **(text or {})}
    font = s["font"] = s["font"] or max(s["min_font"], BASE_FONT * s["width"] / 25.4 / REFERENCE_WIDTH)
    s["size"] = (s["width"] / 25.4, (s["height"] or aspect * s["width"]) / 25.4)
    k = font / BASE_FONT
    rc = {"font.size": font, "axes.titlesize": 1.1 * font, "axes.labelsize": font, "xtick.labelsize": 0.9 * font, "ytick.labelsize": 0.9 * font,
          "legend.fontsize": 0.9 * font, "axes.linewidth": 0.8 * k, "patch.linewidth": k, "lines.linewidth": 1.5 * k,
          "xtick.major.width": 0.8 * k, "ytick.major.width": 0.8 * k, "xtick.minor.width": 0.6 * k, "ytick.minor.width": 0.6 * k,
          "xtick.major.size": 3.5 * k, "ytick.major.size": 3.5 * k, "xtick.minor.size": 2 * k, "ytick.minor.size": 2 * k,
          "xtick.major.pad": 3.5 * k, "ytick.major.pad": 3.5 * k, "axes.labelpad": 4 * k, "axes.titlepad": 6 * k, "axes.formatter.use_mathtext": True}
    return s, t, k, rc


def layer_colors(s, n):
    """n colours from the style's layer_colors gradient."""
    return LinearSegmentedColormap.from_list("layers", s["layer_colors"])(np.linspace(0, 1, n))


def signed_threshold(threshold, values, reach=5):
    """The threshold on the side of the larger magnitude of values (a flux can be negative); None without a threshold or when it is beyond `reach` times the largest magnitude."""
    if threshold is None:
        return None
    return (1 if values.max() >= -values.min() else -1) * threshold if threshold <= reach * np.abs(values).max() else None
