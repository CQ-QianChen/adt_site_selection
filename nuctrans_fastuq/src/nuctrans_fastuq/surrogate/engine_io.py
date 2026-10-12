"""
engine_io.py - Save a trained BayesValidRox Engine as a plain-data HDF5 file, and load it back.

    save_engine(engine, path, case, engine_name, output_qoi, settings=None)
    load_engine(path) -> Engine

The common part (attributes, coordinates, inputs, scaling, training data) is written for every engine. The part
that depends on the surrogate type is written by a function from _TYPE_WRITERS, so
a new type (gpe, gpe_torch) is one more function and one more entry; loading is the
same (_TYPE_READERS, and the build_* functions at the end of this file).
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..bayesvalidrox import Engine, ExpDesigns, Input, PCE, PyLinkForwardModel
from ..bayesvalidrox.surrogate_models.input_space import InputSpace
from ..bayesvalidrox.surrogate_models.meta_model import MetaModel

FORMAT_VERSION = 1
STR = h5py.string_dtype()


def _strings(group, name, values):
    group.create_dataset(name, data=np.array([str(v) for v in values], dtype=object), dtype=STR)


def _float_or_nan(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pack(group, name, blocks, dtype, offsets_name):
    """Store a list of 1-D blocks (one per time step) as one dataset plus offsets."""
    sizes = [np.size(b) for b in blocks]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    flat = np.concatenate([np.ravel(b) for b in blocks]).astype(dtype)
    group.create_dataset(name, data=flat)
    if offsets_name not in group:
        group.create_dataset(offsets_name, data=offsets)
    elif not np.array_equal(group[offsets_name][()], offsets):
        raise ValueError(f"{name}: offsets differ from {offsets_name}")


# ---------------------------------------------------------------- generic tree (plain data)
#
# Nested dicts, lists, arrays and scalars as HDF5 groups and datasets, with the python type
# of every node in the attribute "t" so it comes back as the same type. Children are named
# by index (i0000 ...); a dict key is stored on its child as attributes "key" and "key_t".
# Anything else (class instances, functions, object arrays) raises: nothing is pickled.

_KEY_TYPES = {"str": str, "int": int, "float": float}
_SIMPLE_LISTS = {float: "float", int: "int"}  # lists of only these are stored as one dataset


def write_tree(parent, name, obj, path="", label=None):
    """Write obj as parent[name]. Raises TypeError, naming the path, for unsupported types."""
    path = f"{path}/{label or name}"
    if obj is None:
        node = parent.create_dataset(name, data=0)
        node.attrs["t"] = "none"
    elif isinstance(obj, (bool, np.bool_)) or (isinstance(obj, np.generic) and obj.dtype.kind in "iufc"):
        node = parent.create_dataset(name, data=obj)
        node.attrs["t"] = "np:" + obj.dtype.name if isinstance(obj, np.generic) else "bool"
    elif isinstance(obj, (int, float)):
        node = parent.create_dataset(name, data=obj)
        node.attrs["t"] = type(obj).__name__
    elif isinstance(obj, str):
        node = parent.create_dataset(name, data=obj, dtype=STR)
        node.attrs["t"] = "str"
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind == "U":
            node = parent.create_dataset(name, data=obj.astype(object), dtype=STR)
        elif obj.dtype.kind in "biufc":
            node = parent.create_dataset(name, data=obj)
        else:
            raise TypeError(f"{path}: arrays of dtype {obj.dtype} are not supported")
        node.attrs["t"] = "ndarray"
    elif isinstance(obj, (list, tuple)):
        kind = "list" if isinstance(obj, list) else "tuple"
        simple = {type(v) for v in obj}
        if len(obj) and len(simple) == 1 and simple.issubset(_SIMPLE_LISTS):
            node = parent.create_dataset(name, data=np.array(obj))
            node.attrs["t"] = f"{kind}:{_SIMPLE_LISTS[simple.pop()]}"
        else:
            node = parent.create_group(name)
            node.attrs["t"] = kind
            for i, v in enumerate(obj):
                write_tree(node, f"i{i:04d}", v, path, label=str(i))
    elif isinstance(obj, dict):
        node = parent.create_group(name)
        node.attrs["t"] = "dict"
        if type(obj) is not dict:
            node.attrs["cls"] = type(obj).__name__
        for i, (k, v) in enumerate(obj.items()):
            if type(k).__name__ not in _KEY_TYPES:
                raise TypeError(f"{path}: dict key {k!r} of type {type(k).__name__} is not supported")
            write_tree(node, f"i{i:04d}", v, path, label=str(k))
            node[f"i{i:04d}"].attrs["key"] = str(k) if type(k) is str else repr(k)
            node[f"i{i:04d}"].attrs["key_t"] = type(k).__name__
    else:
        raise TypeError(f"{path}: type {type(obj).__name__} is not supported")


def _read_scalar(raw):
    return raw.decode() if isinstance(raw, bytes) else raw


def read_tree(node, dict_types=None):
    """Read a node written by write_tree. dict_types maps a stored dict class name to a
    factory (for example {"AutoVivification": MetaModel.AutoVivification}); the default is dict."""
    dict_types = dict_types or {}
    kind = node.attrs["t"]
    if kind == "packed":
        return read_packed(node, dict_types)
    if kind == "none":
        return None
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if kind == "str":
            return _read_scalar(value)
        if kind == "bool":
            return bool(value)
        if kind in ("int", "float"):
            return {"int": int, "float": float}[kind](value)
        if kind.startswith("np:"):
            return np.dtype(kind[3:]).type(value)
        if kind == "ndarray":
            if value.dtype.kind == "O":
                return np.array([_read_scalar(v) for v in value.ravel()]).reshape(value.shape)
            return np.asarray(value)
        base, _, item = kind.partition(":")
        return (list if base == "list" else tuple)(value.tolist())
    children = [node[k] for k in sorted(node)]
    if kind in ("list", "tuple"):
        items = [read_tree(c, dict_types) for c in children]
        return items if kind == "list" else tuple(items)
    out = dict_types.get(node.attrs.get("cls"), dict)()
    for c in children:
        key_t = c.attrs["key_t"]
        key = c.attrs["key"] if key_t == "str" else _KEY_TYPES[key_t](c.attrs["key"])
        out[key] = read_tree(c, dict_types)
    return out


# ---------------------------------------------------------------- packed tree (few objects)
#
# The same python objects as write_tree, but stored in a few datasets instead of one HDF5
# object per value: a JSON description of the structure, and every number or string packed
# into a pool that the structure points into by offset. There is one pool per numpy dtype
# (pool_float64, pool_float128, pool_int64, ...), so no value is ever cast. For bookkeeping
# with thousands of tiny values (one HDF5 object costs ~640 bytes).
# read_tree reads it too (the group attribute t is "packed").

_POOL_KINDS = "fiub"  # numpy dtype kinds stored in pools; strings go to the "strings" pool


class _Pools:
    def __init__(self):
        self.parts, self.size = {}, {}

    def add(self, pool, values):
        """Append a 1-D sequence to a pool, return its offset."""
        offset = self.size.get(pool, 0)
        self.parts.setdefault(pool, []).append(values)
        self.size[pool] = offset + len(values)
        return offset


def _pack_node(obj, pools, path):
    """JSON-able description of obj; the values go into pools. Nodes:
    ["none"], ["bool"|"int"|"float", offset], ["str", offset], ["np", dtype, offset] (numpy scalar),
    ["nd", dtype, shape, offset], ["list"|"tuple", children], ["dict", class, keys, children]."""
    if obj is None:
        return ["none"]
    if isinstance(obj, (np.generic, np.ndarray)):  # before int/float: np.float64 is a float
        arr = np.asarray(obj)
        if arr.dtype.kind == "U":
            offset, dtype = pools.add("strings", [str(v) for v in arr.ravel()]), arr.dtype.str
        elif arr.dtype.kind in _POOL_KINDS:
            offset, dtype = pools.add(arr.dtype.name, arr.ravel()), arr.dtype.name
        else:
            raise TypeError(f"{path}: dtype {arr.dtype} is not supported")
        return ["np", dtype, offset] if isinstance(obj, np.generic) else ["nd", dtype, list(arr.shape), offset]
    if isinstance(obj, (bool, int, float)):  # bool first in the name table: bool is an int
        kind = type(obj).__name__
        return [kind, pools.add({"bool": "bool", "int": "int64", "float": "float64"}[kind], np.array([obj]))]
    if isinstance(obj, str):
        return ["str", pools.add("strings", [obj])]
    if isinstance(obj, (list, tuple)):
        return ["list" if isinstance(obj, list) else "tuple",
                [_pack_node(v, pools, f"{path}/{i}") for i, v in enumerate(obj)]]
    if isinstance(obj, dict):
        keys = []
        for k in obj:
            if type(k).__name__ not in _KEY_TYPES:
                raise TypeError(f"{path}: dict key {k!r} of type {type(k).__name__} is not supported")
            keys.append([k if type(k) is str else repr(k), type(k).__name__])
        cls = None if type(obj) is dict else type(obj).__name__
        return ["dict", cls, keys, [_pack_node(v, pools, f"{path}/{k}") for k, v in obj.items()]]
    raise TypeError(f"{path}: type {type(obj).__name__} is not supported")


def write_packed(parent, name, obj):
    """Write obj as the group parent[name]: dataset structure (JSON) plus one pool per dtype."""
    pools = _Pools()
    structure = _pack_node(obj, pools, f"/{name}")
    g = parent.create_group(name)
    g.attrs["t"] = "packed"
    g.create_dataset("structure", data=json.dumps(structure, separators=(",", ":")), dtype=STR)
    for pool, parts in pools.parts.items():
        if pool == "strings":
            g.create_dataset(pool, data=np.array([x for part in parts for x in part], dtype=object), dtype=STR)
        else:
            g.create_dataset(f"pool_{pool}", data=np.concatenate(parts))


def _unpack_node(node, pools, dict_types):
    kind = node[0]
    if kind == "none":
        return None
    if kind in ("bool", "int", "float"):
        return {"bool": bool, "int": int, "float": float}[kind](
            pools[{"bool": "bool", "int": "int64", "float": "float64"}[kind]][node[1]])
    if kind == "str":
        return pools["strings"][node[1]]
    if kind == "np":
        return np.dtype(node[1]).type(pools["strings" if node[1].startswith("<U") else node[1]][node[2]])
    if kind == "nd":
        _, dtype, shape, offset = node
        part = pools["strings" if dtype.startswith("<U") else dtype][offset:offset + int(np.prod(shape))]
        return np.array(part, dtype=np.dtype(dtype)).reshape(shape)
    if kind in ("list", "tuple"):
        items = [_unpack_node(c, pools, dict_types) for c in node[1]]
        return items if kind == "list" else tuple(items)
    if kind == "dict":
        out = dict_types.get(node[1], dict)()
        for (key, key_t), child in zip(node[2], node[3]):
            k = key if key_t == "str" else _KEY_TYPES[key_t](key)
            out[k] = _unpack_node(child, pools, dict_types)
        return out
    raise ValueError(f"unknown packed node {kind!r}")


def read_packed(node, dict_types=None):
    pools = {k[len("pool_"):]: v[()] for k, v in node.items() if k.startswith("pool_")}
    if "strings" in node:
        pools["strings"] = [_read_scalar(v) for v in node["strings"][()]]
    return _unpack_node(json.loads(_read_scalar(node["structure"][()])), pools, dict_types or {})


# ---------------------------------------------------------------- common part

def _write_seen(g, scaler):
    """n_samples_seen_ with its exact scalar type (np.float64 for StandardScaler, int for MinMaxScaler)."""
    seen = scaler.n_samples_seen_
    g.attrs["n_samples_seen"] = seen
    g.attrs["n_samples_seen_type"] = type(seen).__name__


def _write_scaling(f, config):
    g = f.create_group("scaling")
    g.attrs["config_is_none"] = config is None  # scaling_config None is not the same as {"type": "none"}
    config = config or {"type": "none"}
    kind = config.get("type") or "none"
    g.attrs["type"] = kind
    if kind == "div_by_max":
        g.attrs["max_val"] = float(config["max_val"])
    elif kind == "standardize":
        g.attrs["mean"] = float(config["scaler"].mean_[0])
        g.attrs["scale"] = float(config["scaler"].scale_[0])
        _write_seen(g, config["scaler"])
    elif kind == "minmax":
        # sklearn: transform(x) = x * scale_ + min_
        g.attrs["min"] = float(config["scaler"].min_[0])
        g.attrs["scale"] = float(config["scaler"].scale_[0])
        g.attrs["data_min"] = float(config["scaler"].data_min_[0])
        g.attrs["data_range"] = float(config["scaler"].data_range_[0])  # used by descale_std
        g.attrs["data_max"] = float(config["scaler"].data_max_[0])
        _write_seen(g, config["scaler"])
    elif kind != "none":
        raise ValueError(f"Unknown scaling type: {kind}")


def _write_inputs(f, meta):
    if meta.input_obj.rosenblatt:
        raise NotImplementedError("Rosenblatt transformation is not supported.")
    g = f.create_group("inputs")
    _strings(g, "parameter_names", meta.par_names)
    _strings(g, "poly_types", meta.input_space.poly_types)
    g.create_dataset("bounds", data=np.array(meta.input_space.bound_tuples, dtype=float))
    for i, m in enumerate(meta.input_obj.marginals):
        mg = g.create_group(f"marginal_{i:04d}")
        mg.attrs["name"] = str(m.name)
        if m.dist_type is not None:  # data-based marginals have dist_type None: no attribute
            mg.attrs["dist_type"] = str(m.dist_type)
        mg.attrs["parameters"] = np.asarray(m.parameters if m.parameters is not None else [], dtype=float)
        mg.attrs["moments"] = np.asarray(m.moments if m.moments is not None else [], dtype=float)
        # the lists with their exact element types (python float or np.float64), for an identical rebuild
        write_tree(mg, "exact", {"parameters": m.parameters, "moments": m.moments})
        data = np.asarray(m.input_data if m.input_data is not None else [], dtype=float)
        if data.size:
            mg.create_dataset("input_data", data=data)


# Objects are stored as "all attributes except a skip list": an attribute added later by
# BayesValidRox is then stored automatically, or raises TypeError if it is not plain data,
# instead of being silently lost.
_SPACE_SKIP = {"input_object", "j_dist", "prior_space", "orig_j_dist",  # rebuilt / handled below
               "x", "y", "x_values", "out_names", "x_valid", "y_valid"}  # stored under training/ and extras/validation


def _plain_attrs(obj, skip):
    return {k: v for k, v in vars(obj).items() if k not in skip}


def _space_dict(space):
    """Plain-data snapshot of an InputSpace / ExpDesigns. The chaospy joint distribution is not
    stored (rebuilt from the marginals); the KDE is stored as its data array (it is fully
    determined by it, but the samples are random so they must be kept)."""
    if space.orig_j_dist is not None:
        raise NotImplementedError("orig_j_dist is set (Rosenblatt), not supported.")
    out = _plain_attrs(space, _SPACE_SKIP)
    out["prior_dataset"] = None if space.prior_space is None else np.asarray(space.prior_space.dataset)
    return out


def _write_spaces(f, engine):
    """The engine has two input spaces that differ: meta_model.input_space and exp_design."""
    g = f.create_group("spaces")
    write_tree(g, "meta_model", _space_dict(engine.meta_model.input_space))
    write_tree(g, "exp_design", _space_dict(engine.exp_design))


_META_SKIP = {"input_obj", "input_space", "_polycoeffs", "_clf_poly", "_basis_dict", "_coeffs_dict",
              "regression_dict", "par_names", "out_names", "first_out"}  # stored elsewhere, or rebuilt by PCE()
_ENGINE_SKIP = {"meta_model", "model", "exp_design", "scaling_config", "out_names"}
FIRST_OUT_REF = "_clf_poly/b_1"  # marker: this regressor equals the fitted one under pce/


def _same_attrs(a, b):
    """True if two plain-attribute objects hold equal values (arrays compared exactly, NaN equal)."""
    if vars(a).keys() != vars(b).keys():
        return False
    for k, x in vars(a).items():
        y = vars(b)[k]
        try:
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=np.asarray(x).dtype.kind == "f"):
                return False
        except TypeError:
            if x != y:
                return False
    return True


def _first_out(meta):
    """meta.first_out holds a copy of every fitted regressor (equal to _clf_poly["b_1"]) plus
    small per-fit values. The copy is replaced by a marker, after checking that it is equal."""
    out = {}
    for depth, entries in meta.first_out.items():
        out[depth] = []
        for t, entry in enumerate(entries):
            if not _same_attrs(entry["clf_poly"], meta._clf_poly["b_1"][depth][f"y_{t + 1}"]):
                raise NotImplementedError(f"first_out[{depth}][{t}]['clf_poly'] differs from _clf_poly['b_1']")
            out[depth].append({**entry, "clf_poly": FIRST_OUT_REF})
    return out


def _write_extras(f, engine):
    """Everything not needed to evaluate: bookkeeping, flags, forward-model stub, validation."""
    g = f.create_group("extras")
    meta = {**_plain_attrs(engine.meta_model, _META_SKIP), "first_out": _first_out(engine.meta_model)}
    write_packed(g, "meta_model", meta)
    write_packed(g, "engine", _plain_attrs(engine, _ENGINE_SKIP))
    model = {**vars(engine.model), "output": vars(engine.model.output)}
    write_packed(g, "model", model)
    ed = engine.exp_design
    write_packed(g, "validation", {"x_valid": ed.x_valid, "y_valid": ed.y_valid})


def _write_training(f, engine):
    ed = engine.exp_design
    g = f.create_group("training")
    g.create_dataset("X", data=np.asarray(ed.x, dtype=float))
    gy = g.create_group("y")
    for i, depth in enumerate(engine.out_names):
        gy.create_dataset(f"depth_{i:04d}", data=np.asarray(ed.y[depth], dtype=float))


# ---------------------------------------------------------------- PCE part

def _write_pce(f, engine):
    meta = engine.meta_model
    if meta.meta_model_type != "aPCE":
        raise NotImplementedError(f"PCE type '{meta.meta_model_type}' is not supported, only 'aPCE'.")
    if meta.dim_red_method.lower() != "no":
        raise NotImplementedError(f"dim_red_method '{meta.dim_red_method}' is not supported.")

    g = f.create_group("pce")
    g.attrs["meta_model_type"] = meta.meta_model_type
    g.attrs["reg_method"] = str(meta.pce_reg_method)
    g.attrs["bootstrap_method"] = str(meta.bootstrap_method)
    g.attrs["n_bootstrap_itrs"] = int(meta.n_bootstrap_itrs)
    g.attrs["pce_deg"] = np.asarray(meta.pce_deg, dtype=np.int64)
    g.attrs["q_norm"] = np.asarray(meta.pce_q_norm, dtype=float)
    g.attrs["dim_red_method"] = str(meta.dim_red_method)
    g.attrs["input_transform"] = str(meta.input_transform)

    gp = g.create_group("polycoeffs")
    for j in range(len(meta.par_names)):
        gp.create_dataset(f"p_{j:04d}", data=np.asarray(meta._polycoeffs[f"p_{j + 1}"], dtype=float))

    # One group per bootstrap key. The depth axis is folded into the time axis: the entries are the
    # (depth, time step) pairs, depth-major, and the offsets index them (n_depths * n_time + 1).
    n_time = len(engine.exp_design.x_values)
    pairs = [(depth, f"y_{t + 1}") for depth in engine.out_names for t in range(n_time)]
    for b_key in meta._basis_dict:
        gb = g.create_group(b_key)
        basis = [np.asarray(meta._basis_dict[b_key][d][k], dtype=np.int64) for d, k in pairs]
        gb.create_dataset("basis", data=np.vstack(basis))
        gb.create_dataset("term_offsets", data=np.concatenate(
            [[0], np.cumsum([b.shape[0] for b in basis])]).astype(np.int64))
        gb.create_dataset("coeffs", data=np.concatenate(
            [np.ravel(meta._coeffs_dict[b_key][d][k]) for d, k in pairs]).astype(float))
        if b_key in meta._clf_poly:
            _write_regressors(gb, [meta._clf_poly[b_key][d][k] for d, k in pairs])
    _write_regressor_extras(g, meta)


def _write_regressors(gb, clfs):
    gb.create_dataset("reg_intercept", data=np.array([float(c.intercept_) for c in clfs]))
    gb.create_dataset("reg_alpha", data=np.array([float(c.alpha_) for c in clfs]))
    gb.create_dataset("reg_var_y", data=np.array([bool(c.var_y) for c in clfs]))
    gb.create_dataset("reg_normalize", data=np.array([bool(c.normalize) for c in clfs]))
    _pack(gb, "reg_x_mean", [c._x_mean_ for c in clfs], float, "feat_offsets")
    _pack(gb, "reg_x_std", [c._x_std for c in clfs], float, "feat_offsets")
    _pack(gb, "reg_active", [c.active_ for c in clfs], bool, "feat_offsets")
    # coef_ has one entry per active term, so it shares term_offsets with basis
    gb.create_dataset("reg_coef", data=np.concatenate([np.ravel(c.coef_) for c in clfs]).astype(float))
    _pack(gb, "reg_sigma", [c.sigma_ for c in clfs], float, "sigma_offsets")


_REG_SKIP = {"coef_", "sigma_", "active_", "_x_mean_", "_x_std"}  # large arrays, stored as reg_* above


REG_CLASS_KEY = "__class__"  # in regressor_extras: class name of the fitted regressor
_REG_CLASSES = {"RegressionFastARD", "RegressionFastARDExtended"}  # the ones the reg_* arrays describe


def _regressor_extras(clf):
    name = type(clf).__name__
    if name not in _REG_CLASSES:
        raise TypeError(f"Regressor class '{name}' is not supported (only {sorted(_REG_CLASSES)}).")
    return {REG_CLASS_KEY: name, **_plain_attrs(clf, _REG_SKIP)}


def _write_regressor_extras(g, meta):
    """All other attributes of the fitted regressors (settings, diagnostics, exact scalar types)."""
    extras = {b_key: {depth: [_regressor_extras(c) for c in (d[f"y_{t + 1}"] for t in range(len(d)))]
                      for depth, d in per_depth.items()}
              for b_key, per_depth in meta._clf_poly.items()}
    write_packed(g, "regressor_extras", extras)


_TYPE_WRITERS = {"pce": _write_pce}


def surrogate_type_of(engine):
    """Name of the surrogate type of an Engine, as used in the file."""
    name = type(engine.meta_model).__name__
    if name == "PCE":
        return "pce"
    raise NotImplementedError(f"Surrogate class '{name}' is not supported yet.")


# ---------------------------------------------------------------- entry point

def save_engine(engine, path, case, engine_name, output_qoi, settings=None):
    """Write the engine to an HDF5 file at path.

    Written to path + ".tmp" and renamed at the end: if the writer raises (an engine it cannot store),
    no file is left behind and an existing file at path is kept.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        _save(engine, tmp, case, engine_name, output_qoi, settings)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _save(engine, path, case, engine_name, output_qoi, settings):
    kind = surrogate_type_of(engine)
    meta = engine.meta_model
    with h5py.File(path, "w", libver="latest") as f:  # compact object headers, needs HDF5 >= 1.10
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["case"] = str(case)
        f.attrs["engine_name"] = str(engine_name)
        f.attrs["surrogate_type"] = kind
        f.attrs["output_qoi"] = str(output_qoi)
        f.attrs["settings_json"] = json.dumps(settings or {}, default=str)

        g = f.create_group("coordinates")
        g.create_dataset("time", data=np.asarray(engine.exp_design.x_values, dtype=float))
        _strings(g, "depths", engine.out_names)
        g.create_dataset("depths_z", data=np.array([_float_or_nan(d) for d in engine.out_names]))

        _write_inputs(f, meta)
        _write_scaling(f, engine.scaling_config)
        _write_spaces(f, engine)
        _write_extras(f, engine)
        _write_training(f, engine)
        _TYPE_WRITERS[kind](f, engine)


# ---------------------------------------------------------------- reader (plain data)

_REQUIRED_GROUPS = ("coordinates", "inputs", "scaling", "spaces", "extras", "training")


def _attr(v):
    """h5py attribute value as a plain python value or numpy array."""
    return v.item() if isinstance(v, np.generic) else v


def _read_strings(ds):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in ds[()]]


def _blocks(flat, offsets):
    return [flat[offsets[t]:offsets[t + 1]] for t in range(len(offsets) - 1)]


def _read_pce(g, depths, n_params, n_time):
    """Per depth and bootstrap key a list with one dict per time step."""
    out = {"attrs": {k: _attr(v) for k, v in g.attrs.items()},
           "polycoeffs": [g[f"polycoeffs/p_{j:04d}"][()] for j in range(n_params)],
           "fits": {}}
    out["fits"] = {depth: {} for depth in depths}
    for b_key in (k for k in g if k.startswith("b_")):
        d = {k: v[()] for k, v in g[b_key].items()}
        to = d["term_offsets"]
        steps = [{"basis": a, "coeffs": c} for a, c in
                 zip(np.split(d["basis"], to[1:-1]), _blocks(d["coeffs"], to))]
        if "reg_alpha" in d:
            fo, so = d["feat_offsets"], d["sigma_offsets"]
            reg = {k: _blocks(d[f"reg_{k}"], fo) for k in ("x_mean", "x_std", "active")}
            reg["coef"] = _blocks(d["reg_coef"], to)
            reg["sigma"] = [s.reshape(k, k) for s, k in zip(_blocks(d["reg_sigma"], so), np.diff(to))]
            for n, step in enumerate(steps):
                step["reg"] = {k: v[n] for k, v in reg.items()}
                for k in ("intercept", "alpha", "var_y", "normalize"):
                    step["reg"][k] = d[f"reg_{k}"][n]
        for i, depth in enumerate(depths):  # the entries are depth-major: n_time per depth
            out["fits"][depth][b_key] = steps[i * n_time:(i + 1) * n_time]
    if "regressor_extras" in g:
        for b_key, per_depth in read_tree(g["regressor_extras"]).items():
            for depth, entries in per_depth.items():
                for step, extra in zip(out["fits"][depth][b_key], entries):
                    step["reg"]["extra"] = extra
    return out


_TYPE_READERS = {"pce": _read_pce}


def _read_file(path, dict_types=None):
    """Read engine.h5 into plain dicts and arrays (the first half of load_engine).

    Raises on an unknown format version, a missing group or an unsupported surrogate type.
    dict_types: see read_tree (for example to get AutoVivification back).
    """
    with h5py.File(path, "r") as f:
        if f.attrs.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {f.attrs.get('format_version')}, expected {FORMAT_VERSION}")
        kind = f.attrs["surrogate_type"]
        if kind not in _TYPE_READERS or kind not in f:
            raise ValueError(f"{path}: surrogate type '{kind}' is not supported or its group is missing")
        missing = [g for g in _REQUIRED_GROUPS if g not in f]
        if missing:
            raise ValueError(f"{path}: missing groups {missing}")

        depths = _read_strings(f["coordinates/depths"])
        names = _read_strings(f["inputs/parameter_names"])
        marginals = []
        for j in range(len(names)):
            mg = f[f"inputs/marginal_{j:04d}"]
            marginals.append({"dist_type": None, **{k: _attr(v) for k, v in mg.attrs.items()},
                              **read_tree(mg["exact"]),  # parameters, moments with their exact types
                              "input_data": mg["input_data"][()] if "input_data" in mg else None})
        return {
            "attrs": {k: _attr(v) for k, v in f.attrs.items()},
            "time": f["coordinates/time"][()],
            "depths": depths,
            "depths_z": f["coordinates/depths_z"][()],
            "parameter_names": names,
            "poly_types": _read_strings(f["inputs/poly_types"]),
            "bounds": f["inputs/bounds"][()],
            "marginals": marginals,
            "scaling": {k: _attr(v) for k, v in f["scaling"].attrs.items()},
            "spaces": {k: read_tree(v, dict_types) for k, v in f["spaces"].items()},
            "extras": {k: read_tree(v, dict_types) for k, v in f["extras"].items()},
            "train_X": f["training/X"][()],
            "train_y": {d: f[f"training/y/depth_{i:04d}"][()] for i, d in enumerate(depths)},
            kind: _TYPE_READERS[kind](f[kind], depths, len(names), len(f["coordinates/time"])),
        }


# ---------------------------------------------------------------- loader (rebuilds the Engine)

def build_input(data):
    """bayesvalidrox Input with one marginal per parameter."""
    inputs = Input()
    for m in data["marginals"]:
        inputs.add_marginals(name=m["name"], dist_type=m["dist_type"],
                             parameters=m["parameters"], input_data=m["input_data"])
        inputs.marginals[-1].moments = m["moments"]
    return inputs


def fill_space(space, stored, inputs):
    """Give an InputSpace / ExpDesigns the stored values, with the distributions rebuilt.

    init_param_space is not called: it draws random samples and would recompute the bounds
    and moments from them. The stored values are used instead, and the KDE is rebuilt from
    its stored data array (identical to the original).
    """
    space.j_dist, poly_types = space.build_polytypes(rosenblatt=inputs.rosenblatt)
    if list(poly_types) != list(stored["poly_types"]):
        raise ValueError(f"poly_types rebuilt as {poly_types}, file has {stored['poly_types']}")
    for name, value in stored.items():
        if name != "prior_dataset":
            setattr(space, name, value)
    space.prior_space = None if stored["prior_dataset"] is None else gaussian_kde(stored["prior_dataset"])
    return space


def build_input_space(data, inputs):
    """The InputSpace of the meta model."""
    stored = data["spaces"]["meta_model"]
    return fill_space(InputSpace(inputs, meta_model_type=stored["meta_model_type"]), stored, inputs)


def build_exp_design(data, inputs):
    """ExpDesigns with the training data and its own input space (not the same as the meta model's)."""
    stored = data["spaces"]["exp_design"]
    settings = {k: stored[k] for k in ("sampling_method", "hdf5_file", "out_dir", "n_init_samples", "n_new_samples",
                                       "n_max_samples", "explore_method", "exploit_method", "util_func",
                                       "tradeoff_scheme", "n_candidates", "n_cand_groups", "max_func_itr")}
    design = fill_space(ExpDesigns(inputs, meta_model_type=stored["meta_model_type"], **settings), stored, inputs)
    design.x, design.y = data["train_X"], data["train_y"]
    design.x_values = data["time"]
    design.out_names = list(data["depths"])
    return design


def _regressor(r, classes):
    extra = dict(r["extra"])
    clf = classes[extra.pop(REG_CLASS_KEY)]()
    clf.coef_, clf.sigma_, clf.active_ = r["coef"], r["sigma"], r["active"]
    clf._x_mean_, clf._x_std = r["x_mean"], r["x_std"]
    clf.alpha_, clf.intercept_ = float(r["alpha"]), float(r["intercept"])
    clf.var_y, clf.normalize = bool(r["var_y"]), bool(r["normalize"])
    for name, value in extra.items():  # settings, diagnostics, and the exact scalar types
        setattr(clf, name, value)
    return clf


def build_pce(data, inputs=None):
    """PCE meta model with the fitted dictionaries filled in, ready for eval_metamodel."""
    a = data["pce"]["attrs"]
    inputs = inputs if inputs is not None else build_input(data)
    meta = PCE(inputs, meta_model_type=a["meta_model_type"], pce_reg_method=a["reg_method"],
               bootstrap_method=a["bootstrap_method"], n_bootstrap_itrs=a["n_bootstrap_itrs"],
               pce_deg=a["pce_deg"], pce_q_norm=a["q_norm"], dim_red_method=a["dim_red_method"],
               input_transform=a["input_transform"])
    classes = {c.__name__: c for c in meta.regression_dict.values()}
    meta.par_names = list(data["parameter_names"])
    meta.out_names = list(data["depths"])
    meta._polycoeffs = {f"p_{j + 1}": p for j, p in enumerate(data["pce"]["polycoeffs"])}
    meta.input_space = build_input_space(data, inputs)

    meta._basis_dict, meta._coeffs_dict, meta._clf_poly = (MetaModel.AutoVivification() for _ in range(3))
    for depth, fits in data["pce"]["fits"].items():
        for b_key, steps in fits.items():
            for t, step in enumerate(steps):
                meta._basis_dict[b_key][depth][f"y_{t + 1}"] = step["basis"]
                meta._coeffs_dict[b_key][depth][f"y_{t + 1}"] = step["coeffs"]
                if "reg" in step:
                    meta._clf_poly[b_key][depth][f"y_{t + 1}"] = _regressor(step["reg"], classes)
    for name, value in data["extras"]["meta_model"].items():
        setattr(meta, name, value)
    for depth, entries in meta.first_out.items():  # the marker stands for a copy of the fitted regressor
        for t, entry in enumerate(entries):
            if entry["clf_poly"] != FIRST_OUT_REF:
                raise ValueError(f"unknown first_out reference {entry['clf_poly']!r}")
            entry["clf_poly"] = _regressor(data["pce"]["fits"][depth]["b_1"][t]["reg"], classes)
    return meta


_SEEN_TYPES = {"int": int, "float": float, "int64": np.int64, "float64": np.float64}


def build_scaling(sc):
    """Scaling config in the form of scaling.py (fit_scaling), from the file's scaling attributes."""
    if sc["config_is_none"]:
        return None
    config = {"type": sc["type"]}
    if sc["type"] == "div_by_max":
        config["max_val"] = float(sc["max_val"])
    elif sc["type"] == "standardize":
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = np.array([sc["mean"]]), np.array([sc["scale"]])
        scaler.var_, scaler.n_features_in_ = scaler.scale_ ** 2, 1
        scaler.n_samples_seen_ = _SEEN_TYPES[sc["n_samples_seen_type"]](sc["n_samples_seen"])
        config["scaler"] = scaler
    elif sc["type"] == "minmax":
        scaler = MinMaxScaler()
        scaler.min_, scaler.scale_ = np.array([sc["min"]]), np.array([sc["scale"]])
        scaler.data_min_, scaler.data_range_ = np.array([sc["data_min"]]), np.array([sc["data_range"]])
        scaler.data_max_ = np.array([sc["data_max"]])
        scaler.n_samples_seen_ = _SEEN_TYPES[sc["n_samples_seen_type"]](sc["n_samples_seen"])
        scaler.n_features_in_ = 1
        config["scaler"] = scaler
    return config


def load_engine(path):
    """Load engine.h5 and rebuild the BayesValidRox Engine, ready to evaluate."""
    return build_engine(_read_file(path, dict_types={"AutoVivification": MetaModel.AutoVivification}))


def build_engine(data):
    """Engine with the trained meta model, the training data and the scaling, ready to evaluate."""
    kind = data["attrs"]["surrogate_type"]
    if kind != "pce":
        raise NotImplementedError(f"surrogate_type '{kind}' is not supported yet")
    inputs = build_input(data)
    meta = build_pce(data, inputs)

    exp_design = build_exp_design(data, inputs)

    model = PyLinkForwardModel()  # never run, the Engine only reads the output names
    for name, value in data["extras"]["model"].items():
        if name == "output":
            vars(model.output).update(value)
        else:
            setattr(model, name, value)
    engine = Engine(meta, model, exp_design)
    engine.start_engine()
    for name, value in data["extras"]["engine"].items():
        setattr(engine, name, value)
    exp_design.x_valid = data["extras"]["validation"]["x_valid"]
    exp_design.y_valid = data["extras"]["validation"]["y_valid"]
    engine.scaling_config = build_scaling(data["scaling"])
    return engine
