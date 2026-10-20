import re
from pathlib import Path

import h5py
import numpy as np
import yaml


def read_uq(path):
    """A propagation or sensitivity file as a flat dict: attributes by name, datasets by path."""
    data = {}

    def add(name, obj):
        if isinstance(obj, h5py.Dataset):
            data[name] = obj.asstr()[()] if obj.dtype.kind == "O" else obj[()]

    with h5py.File(path, "r") as f:
        data.update(f.attrs)
        f.visititems(add)
    return data


def depth_labels(data):
    return [f"{z:.2f} m" if np.isfinite(z) else name for z, name in zip(data["axes/depth_z"], data["axes/depth"])]


def depth_index(data, depth):
    """Index and label of a depth given by name or in m (nearest)."""
    names, z = list(data["axes/depth"]), data["axes/depth_z"]
    i = names.index(depth) if isinstance(depth, str) else int(np.abs(z - depth).argmin())
    return i, names[i].replace("_", " ") if np.isnan(z[i]) else f"{z[i]:.0f} m"


def time_label(t):
    return f"t = {t:.0e} yr".replace("e+0", "e")


def read_stratigraphy(path):
    """(name, top, bottom) per layer from a rock_interface.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        interfaces = {name: entry[0]["value"] for name, entry in yaml.safe_load(f).items()}
    names = [name[:-len("_bottom")] for name in interfaces if name.endswith("_bottom")]
    tops = [interfaces[names[0] + "_top"]] + [interfaces[name + "_bottom"] for name in names[:-1]]
    return [(name, top, interfaces[name + "_bottom"]) for name, top in zip(names, tops)]


def parameter_labels(names):
    """Readable labels, e.g. rock_data_Aptian_porosity -> Aptian porosity."""
    labels = []
    for name in names:
        nuclide = re.fullmatch(r"nuclide_(?:water_)?(.+?)_data_(.+)_[A-Za-z]+-\d+", name)
        labels.append(f"{nuclide[2]} {nuclide[1]}".replace("_", " ") if nuclide
                      else re.sub(r"^(geometry_rock_interface|rock_data)_", "", name).replace("_", " "))
    return labels


def parameter_units(names, folders):
    """Unit of each parameter, read from the data folders (geometry, rock_data, ...): a name is <folder>_<file>_<key>, the unit is unit_str of <key> in <folder>/<file>.yaml; "" when unknown or without unit."""
    superscripts = {"^2": "²", "^3": "³", "^-1": "⁻¹", "^-2": "⁻²"}
    units = {}
    for name in names:
        units[name] = ""
        for folder in map(Path, folders):
            if not name.startswith(folder.name + "_"):
                continue
            rest = name[len(folder.name) + 1:]
            for file in sorted(folder.glob("*.yaml"), key=lambda f: -len(f.stem)):
                if rest.startswith(file.stem + "_"):
                    with open(file, "r", encoding="utf-8") as f:
                        entry = yaml.safe_load(f).get(rest[len(file.stem) + 1:])
                    unit = (entry[0] if isinstance(entry, list) else entry or {}).get("unit_str") or ""
                    units[name] = re.sub(r"\^-?\d", lambda m: superscripts.get(m[0], m[0]), unit)
                    break
    return units


def parameter_rows(names, layers, property_order=("density", "porosity", "diffusivity")):
    """(label, index) per parameter, by layer from the top, then by property_order; a layer bottom is the interface to the next layer."""
    layer_names = [name.replace("_", " ") for name, _, _ in layers]

    def key(row):
        layer, _, prop = row[0].rpartition(" ")
        return layer_names.index(layer) if layer in layer_names else len(layer_names), property_order.index(prop) if prop in property_order else len(property_order)

    after = dict(zip(layer_names, layer_names[1:]))
    rows = sorted(zip(parameter_labels(names), range(len(names))), key=key)
    return [(f"{label[:-7]}/{after[label[:-7]]} interface" if label.endswith(" bottom") and label[:-7] in after else label, i) for label, i in rows]
