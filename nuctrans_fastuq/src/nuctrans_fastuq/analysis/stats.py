import numpy as np

from .io import depth_index

HEADER = ("scenario", "engine_name", "surrogate_type", "output_qoi", "qoi_unit", "n_samples", "seed", "created")


def key_numbers(propagation, sensitivity, rows, labels=None, threshold=None, levels=None, drivers=3):
    """Per labelled depth (default: the one with the largest peak): the peak over time of every sample, as median and 5-95% range of its magnitude; the main inputs by total Sobol index averaged over time, weighted by the output variance, with the direction of their effect; the exceedance of the threshold."""
    if list(propagation["axes/parameter_names"]) != list(sensitivity["axes/parameter_names"]):
        raise ValueError("propagation and sensitivity files have different parameters")
    if threshold is not None and levels is None:
        raise ValueError("levels are needed with a threshold")
    time, output, X = propagation["axes/time"], propagation["output/mean"], propagation["input/X"]
    names = {j: label for label, j in rows}
    peaks = np.take_along_axis(output, np.abs(output).argmax(axis=1)[:, None, :], axis=1)[:, 0, :]  # signed value at each sample's largest magnitude
    if labels:
        picks = [(name, depth_index(propagation, key)[0]) for key, name in labels.items()]
    else:
        i = int(np.median(np.abs(peaks), axis=0).argmax())
        z = propagation["axes/depth_z"][i]
        picks = [(depth_index(propagation, propagation["axes/depth"][i] if np.isnan(z) else z)[1], i)]
    depths = {}
    for label, i in picks:
        y, peak = output[:, :, i], peaks[:, i]
        sign = 1 if np.median(peak) >= 0 else -1
        low, median, high = sign * np.percentile(np.abs(peak), [5, 50, 95])
        weight = y.var(axis=0)
        st = (np.nan_to_num(sensitivity["ST"][:, :, i]) * weight).sum(axis=1) / weight.sum() if weight.sum() > 0 else np.zeros(len(names))
        top = [j for j in np.argsort(st)[::-1][:drivers] if st[j] > 0]
        depths[label] = {"peak": {"median": median, "low": low, "high": high}, "drivers": [
            {"label": names[j], "st": st[j], "direction": int(np.sign(np.nan_to_num(np.corrcoef(X[:, j], peak)[0, 1])))} for j in top]}
        if threshold is not None:
            size = np.abs(y)
            first = lambda curve: float(time[(curve > threshold).argmax()]) if (curve > threshold).any() else None  # noqa: E731
            share = float((np.abs(peak) > threshold).mean())
            depths[label]["threshold"] = {"value": threshold, "share": share, "median_exceeds": first(np.median(size, axis=0)), "p95_exceeds": first(np.percentile(size, 95, axis=0)),
                                          "level": "high" if share > levels[1] else "medium" if share >= levels[0] else "low"}
    return {"header": {k: propagation[k] for k in HEADER}, "period": time[-1], "headline": next(iter(depths)), "depths": depths}
