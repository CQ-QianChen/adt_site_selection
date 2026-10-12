"""
data_io.py - Read the model box's HDF5 files into X and y for the surrogate.

    X, y, param_names, t_values = load_and_extract_data(path, output_qoi, z_interest)

File layout:

    sampled_data/uncertain_parameters/{folder}/{layer}/{property} -> (n_samples,)
    simulation_results/{qoi}/keys   -> (n_points, 4), rows are (x, y, z, t)
    simulation_results/{qoi}/values -> (n_points, n_samples)
                                       (n_points, n_samples, n_comp) for Flux

keys and values are parallel arrays, ordered all depths for t0, then t1, ...
Only 1D (x = y = 0) is supported. Depths are picked by nearest node, not
interpolated, so values from two layers are never mixed.

A file without simulation_results gives y = None.
"""

from pathlib import Path

import h5py
import numpy as np

# OGS time is in seconds; the surrogate box uses years (365.25 days)
SECONDS_PER_YEAR = 31_557_600.0
# 1 m³ = 1000 L: converts (mol/L)*(m/s) -> mol/(m²*year) when multiplied with SECONDS_PER_YEAR
LITERS_PER_M3 = 1000.0
FLUX_TO_ANNUAL_M2 = LITERS_PER_M3 * SECONDS_PER_YEAR

# Group order; fixes the column order of X
GROUP_ORDER = (
    "geometry",
    "rock_data",
    "nuclide_sorption_data",
    "nuclide_water_diffusivity_data",
)

DESIGN_KEY = "sampled_data"
RESPONSE_KEY = "simulation_results"
UNCERTAIN_KEY = "uncertain_parameters"


# --------------------------------------------------------------------------- #
# design (inputs)
# --------------------------------------------------------------------------- #
def _walk_design(group):
    """Yield (param_name, values) in a fixed order.

    param_name is "{folder}_{layer}_{property}", e.g. rock_data_Host_rock_porosity.
    Groups in GROUP_ORDER, layers and properties sorted.
    """
    names = list(group.keys())
    ordered = [g for g in GROUP_ORDER if g in names]
    ordered += sorted(n for n in names if n not in GROUP_ORDER)  # unknown groups last

    for folder in ordered:
        layers = group[folder]
        for layer in sorted(layers.keys()):
            props = layers[layer]
            # normally a group of properties, but allow a plain dataset
            if isinstance(props, h5py.Dataset):
                yield f"{folder}_{layer}", np.asarray(props[()]).ravel()
                continue
            for prop in sorted(props.keys()):
                yield f"{folder}_{layer}_{prop}", np.asarray(props[prop][()]).ravel()


def read_design(h5file, param_names=None):
    """X (n_samples, n_params) and the parameter names.

    param_names: column order to use, from
    prepare_data.get_param_names_from_config(). Always pass it for training:
    without it the order is alphabetical, which does not match the marginals,
    and every column would get the wrong distribution without any error.
    """
    if DESIGN_KEY in h5file:
        root = h5file[DESIGN_KEY]
        root = root[UNCERTAIN_KEY] if UNCERTAIN_KEY in root else root
    elif UNCERTAIN_KEY in h5file:
        root = h5file[UNCERTAIN_KEY]
    else:
        raise KeyError(
            f"Neither '{DESIGN_KEY}' nor '{UNCERTAIN_KEY}' found in the file. Available: {list(h5file.keys())}"
        )

    found = dict(_walk_design(root))
    if not found:
        raise ValueError("No uncertain parameters found in the design.")

    if param_names is None:
        names = list(found)
    else:
        missing = [n for n in param_names if n not in found]
        if missing:
            raise KeyError(
                f"Parameters requested by the config are absent from the data "
                f"file: {missing}\nFile contains: {sorted(found)}"
            )
        names = list(param_names)

    X = np.column_stack([found[n] for n in names])
    return X, names


# --------------------------------------------------------------------------- #
# response (outputs)
# --------------------------------------------------------------------------- #
def _nearest(available, requested):
    """Index of the nearest value (nearest node, no interpolation)."""
    return int(np.argmin(np.abs(np.asarray(available) - float(requested))))


def read_response(h5file, output_qoi, z_interest=None, t_interest=None,
                  flux_component=2):
    """One QoI as {depth_label: array(n_samples, n_timesteps)}.

    z_interest / t_interest: None keeps all.
    flux_component: component of a vector QoI; 2 (z) in 1D.
    Returns (y, t_values), or (None, None) if the file has no results.
    """
    if RESPONSE_KEY not in h5file:
        return None, None

    results = h5file[RESPONSE_KEY]
    if output_qoi not in results:
        raise KeyError(
            f"QoI '{output_qoi}' not in the results. Available: {list(results.keys())}"
        )

    keys = results[output_qoi]["keys"][:]        # (n_points, 4) -> x, y, z, t
    values = results[output_qoi]["values"][:]    # (n_points, n_samples[, n_comp])

    # vector QoI (Flux): keep one component
    if values.ndim == 3:
        values = values[:, :, flux_component]

    x, yc, z, t = keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3] / SECONDS_PER_YEAR
    if not (np.allclose(x, 0.0) and np.allclose(yc, 0.0)):
        raise NotImplementedError(
            "Non-zero x/y found: this file is 2D/3D. read_response() currently "
            "assumes a 1D column (x = y = 0)."
        )

    z_all = np.unique(z)
    t_all = np.unique(t)

    t_keep = t_all if t_interest is None else np.array(
        [t_all[_nearest(t_all, tv)] for tv in np.atleast_1d(t_interest)]
    )
    z_keep = z_all if z_interest is None else np.array(
        [z_all[_nearest(z_all, zv)] for zv in np.atleast_1d(z_interest)]
    )

    n_samples = values.shape[1]
    y = {}
    for z_target in z_keep:
        rows = np.flatnonzero(z == z_target)          # one row per time step
        t_rows = t[rows]
        order = np.argsort(t_rows)
        rows, t_rows = rows[order], t_rows[order]

        cols = [rows[np.flatnonzero(t_rows == tv)[0]] for tv in t_keep]
        # (n_timesteps, n_samples) -> (n_samples, n_timesteps)
        y[str(float(z_target))] = values[cols, :].T.reshape(n_samples, len(t_keep))

    return y, t_keep


def resolve_boundary_z(z_name, sample_geometry):
    """Depth of a boundary name for one sample.

    sample_geometry: {interface_name: depth} of this sample, from its
    geometry_rock_interface_* columns. '<Layer>_top' falls back to the next
    interface above. If z_name is numeric, it is returned directly as float.
    """
    try:
        return float(z_name)
    except (ValueError, TypeError):
        pass

    if z_name in sample_geometry:
        return float(sample_geometry[z_name])

    if str(z_name).endswith("_top"):
        layer_name = str(z_name)[: -len("_top")]
        # top of a layer = next interface above its bottom
        bottom_name = f"{layer_name}_bottom"
        if bottom_name in sample_geometry:
            own_bottom = sample_geometry[bottom_name]
            above = [z for name, z in sample_geometry.items()
                     if name.endswith("_bottom") and z > own_bottom]
            if above:
                return float(min(above))

    raise KeyError(
        f"Cannot resolve boundary '{z_name}' for this sample. "
        f"Available interfaces: {sorted(sample_geometry)}"
    )


def read_response_moving_boundary(h5file, output_qoi, z_interest, param_names, X,
                                  t_interest=None, flux_component=2):
    """Like read_response(), for cases with sampled geometry (Case2).

    Every sample has its own mesh, so one fixed depth is in different layers
    for different samples. For each sample, interpolate its z -> value curve
    at each time step and read it at that sample's own boundary depth, or at
    a fixed numeric depth (e.g. -150.0).

    z_interest: boundary names (e.g. 'Host_rock_top') or numeric depths (e.g. -150.0).
    param_names, X: from read_design(), used to get each sample's geometry.
    Returns (y, t_values) like read_response(), keyed by boundary name or depth string.
    """
    from scipy.interpolate import interp1d

    results = h5file[RESPONSE_KEY]
    if output_qoi not in results:
        raise KeyError(
            f"QoI '{output_qoi}' not in the results. Available: {list(results.keys())}"
        )

    keys = results[output_qoi]["keys"][:]
    values = results[output_qoi]["values"][:]
    if values.ndim == 3:
        values = values[:, :, flux_component]

    x, yc, z, t = keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3] / SECONDS_PER_YEAR
    if not (np.allclose(x, 0.0) and np.allclose(yc, 0.0)):
        raise NotImplementedError(
            "Non-zero x/y found: this file is 2D/3D. "
            "read_response_moving_boundary() currently assumes a 1D column."
        )

    t_all = np.unique(t)
    t_keep = t_all if t_interest is None else np.array(
        [t_all[_nearest(t_all, tv)] for tv in np.atleast_1d(t_interest)]
    )
    n_samples = values.shape[1]

    geometry_prefix = "geometry_rock_interface_"
    geometry_cols = {
        name[len(geometry_prefix):]: i
        for i, name in enumerate(param_names) if name.startswith(geometry_prefix)
    }
    if not geometry_cols:
        raise ValueError(
            "No geometry_rock_interface_* columns found in param_names - "
            "this file's uncertain_parameters has no sampled geometry, so "
            "read_response() (nearest-node, shared depth) is what you want "
            "instead, not read_response_moving_boundary()."
        )

    y = {str(zname): np.full((n_samples, len(t_keep)), np.nan) for zname in z_interest}

    for i in range(n_samples):
        sample_geometry = {name: float(X[i, col]) for name, col in geometry_cols.items()}

        # this sample's mesh: rows that are not NaN
        has_data = ~np.isnan(values[:, i])
        z_sample = z[has_data]
        t_sample = t[has_data]
        v_sample = values[has_data, i]

        for t_target in t_keep:
            t_idx = np.flatnonzero(t_sample == t_target)
            if len(t_idx) < 2:
                continue  # need 2+ points to interpolate
            z_at_t = z_sample[t_idx]
            v_at_t = v_sample[t_idx]
            order = np.argsort(z_at_t)
            f_interp = interp1d(z_at_t[order], v_at_t[order], kind="linear",
                                fill_value="extrapolate")

            t_col = int(np.flatnonzero(t_keep == t_target)[0])
            for zname in z_interest:
                z_val_sample = resolve_boundary_z(zname, sample_geometry)
                y[str(zname)][i, t_col] = float(f_interp(z_val_sample))

    return y, t_keep


# --------------------------------------------------------------------------- #
# load_and_extract_data
# --------------------------------------------------------------------------- #

def _filter_depths(y, z_range, z_min_variance, path, discrete_depths=None):
    """Keep depths inside any interval in z_range, plus any discrete_depths, that vary more than z_min_variance."""
    keys = list(y)
    ranges = []
    if z_range:
        if isinstance(z_range, (list, tuple)) and len(z_range) > 0 and isinstance(z_range[0], (list, tuple)):
            ranges = list(z_range)
        else:
            ranges = [z_range]

    if not ranges and not discrete_depths:
        return y

    kept = set()
    numeric_keys = []
    numeric_depths = []
    for k in keys:
        try:
            val = float(k)
            numeric_keys.append(k)
            numeric_depths.append(val)
        except (TypeError, ValueError):
            kept.add(k)

    # 1. Check intervals
    if ranges:
        for r in ranges:
            if r is None:
                continue
            top, bottom = r
            for k, z in zip(numeric_keys, numeric_depths):
                if (top is None or z <= top) and (bottom is None or z >= bottom):
                    kept.add(k)

    # 2. Check discrete depths
    if discrete_depths and numeric_depths:
        depths_arr = np.array(numeric_depths)
        for zv in discrete_depths:
            try:
                target = float(zv)
                idx = int(np.argmin(np.abs(depths_arr - target)))
                kept.add(numeric_keys[idx])
            except (TypeError, ValueError):
                if str(zv) in y:
                    kept.add(str(zv))

    if not kept:
        raise ValueError(
            f"No depth nodes fall inside requested ranges {ranges} or discrete depths {discrete_depths} "
            f"in {path.name}."
        )

    # 3. Variance check
    if z_min_variance is not None:
        varying = [k for k in kept if np.asarray(y[k]).var(axis=0).max() > z_min_variance]
        dropped = len(kept) - len(varying)
        if not varying:
            raise ValueError(
                f"Every depth selected in {path.name} varies less than "
                f"min_variance={z_min_variance:g}, so none can be fitted. "
                f"Lower min_variance, widen the range, or pick a later "
                f"t_interest."
            )
        if dropped:
            print(f"  z_interest: kept {len(varying)} depths, "
                  f"dropped {dropped} with variance <= {z_min_variance:g}")
        kept = varying

    # Sort descending (surface to deep)
    def _sort_k(k):
        try:
            return (0, -float(k))
        except (TypeError, ValueError):
            return (1, str(k))

    sorted_keys = sorted(kept, key=_sort_k)
    return {k: y[k] for k in sorted_keys}


def load_and_extract_data(file_path, output_qoi, z_interest=None, t_interest=None,
                          transform_config=None, flux_component=2,
                          param_names=None, qoi_floor=None,
                          z_range=None, z_min_variance=None, keep_keys=None):
    """Load an HDF5 file from the model box, formatted for BayesValidRox.
    Args:
        param_names: column order for X, from prepare_data.get_param_names_from_config().
        qoi_floor: numerical noise floor threshold. Values below this threshold
            are set to 0.0 on raw OGS units. For flux, the threshold
            applies to absolute values |y| < qoi_floor to preserve negative fluxes.
            If None, defaults to 1e-20 for flux QoIs and 1e-12 for concentration QoIs.
            Non-zero flux outputs are converted to mol/(m²·year).
        z_range: (top, bottom) depth window, or list of intervals [(top, bottom), ...]; None = no limit.
        z_min_variance: drop depths that vary less than this.
        keep_keys: exact depths to keep for validation.

    Returns:
        X           : np.ndarray (n_samples, n_params)
        y           : dict {depth_label: (n_samples, n_timesteps)}, or None when
                      the file has no model results (sample-only file)
        param_names : list[str], aligned with the columns of X
        t_values    : np.ndarray of the time steps kept, or None
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Normalize z_range: can be a single (top, bottom) or a list of intervals [(top, bottom), ...]
    norm_ranges = []
    if z_range is not None:
        if isinstance(z_range, (list, tuple)) and len(z_range) > 0 and isinstance(z_range[0], (list, tuple)):
            norm_ranges = list(z_range)
        else:
            norm_ranges = [z_range]

    z_list = list(np.atleast_1d(z_interest)) if z_interest is not None else []

    def _is_numeric(val):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    has_geom_params = any(str(p).startswith("geometry_rock_interface_") for p in (param_names or []))
    has_boundary_names = any(isinstance(v, str) and not _is_numeric(v) for v in z_list)
    is_moving_boundary = has_geom_params or has_boundary_names

    print(f"Loading samples from: {path}")
    with h5py.File(path, "r") as f:
        X, param_names = read_design(f, param_names=param_names)
        if RESPONSE_KEY not in f:
            y, t_values = None, None
        elif is_moving_boundary:
            y, t_values = read_response_moving_boundary(
                f, output_qoi, z_list, param_names, X,
                t_interest=t_interest, flux_component=flux_component
            )
        else:
            # If ranges are requested, read all nodes so _filter_depths can slice them.
            # If discrete only, read nearest nodes directly.
            target_z = None if norm_ranges else z_interest
            y, t_values = read_response(
                f, output_qoi, target_z, t_interest, flux_component=flux_component
            )

    if y is not None:
        is_flux = "flux" in output_qoi.lower()
        if qoi_floor is not None:
            threshold = float(qoi_floor)
        elif is_flux:
            threshold = 1e-20  # (mol/L)/(m/s)
        else:
            threshold = 1e-12

        if threshold is not None and threshold > 0:
            for key in y:
                if is_flux:
                    # For flux, threshold using absolute magnitude so negative values are preserved
                    y[key][np.abs(y[key]) < threshold] = 0.0
                else:
                    # For concentration (non-negative), values below threshold are numerical noise
                    y[key][y[key] < threshold] = 0.0

        # Convert flux from (mol/L)*(m/s) to mol/(m²*year) (LITERS_PER_M3 * SECONDS_PER_YEAR)
        if is_flux:
            for key in y:
                y[key] = y[key] * FLUX_TO_ANNUAL_M2

        # depth window + drop flat depths; after the floor, so it sees the floored values
        if keep_keys is not None:
            missing = [k for k in keep_keys if k not in y]
            if missing:
                raise KeyError(
                    f"Depths kept for training are absent from {path.name}: "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            y = {k: y[k] for k in keep_keys}
        elif not is_moving_boundary and (norm_ranges or (z_min_variance is not None and target_z is None)):
            discrete = z_list if norm_ranges else None
            y = _filter_depths(y, norm_ranges, z_min_variance, path, discrete_depths=discrete)

        if transform_config is not None:
            from .scaling import scale_data
            y = scale_data(y, transform_config)

    return X, y, param_names, t_values


def describe(file_path):
    """Print what a results file contains."""
    with h5py.File(file_path, "r") as f:
        print(f"File: {file_path}")
        print(f"Top-level keys: {list(f.keys())}")
        if DESIGN_KEY in f:
            X, names = read_design(f)
            print(f"\nDesign : {X.shape[0]} samples x {X.shape[1]} parameters")
            for n in names:
                print(f"   - {n}")
        if RESPONSE_KEY in f:
            print(f"\nQoIs available: {list(f[RESPONSE_KEY].keys())}")
            for q in f[RESPONSE_KEY]:
                k = f[RESPONSE_KEY][q]["keys"]
                v = f[RESPONSE_KEY][q]["values"]
                z = np.unique(k[:, 2])
                t = np.unique(k[:, 3])
                print(f"   {q}: {len(z)} depths x {len(t)} time steps, "
                      f"values{v.shape}")
        else:
            print("\nNo model results in this file (sample-only).")
