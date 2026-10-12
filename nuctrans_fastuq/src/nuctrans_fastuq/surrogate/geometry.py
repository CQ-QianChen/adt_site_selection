"""
geometry.py - Read a case's layers from geometry/rock_interface.yaml.

The file lists interfaces, not layers:

    Quaternary_top:    value: -0.004
    Quaternary_bottom: value: -50.0
    Albian_bottom:     value: -100.0
    ...

A layer goes from its own *_top (or the previous *_bottom) down to its
*_bottom. For Case2 (sampled geometry) the nominal `value` is used.
"""

from pathlib import Path

import yaml

INTERFACE_FILE = "rock_interface.yaml"

# Layer colours, light at the top to dark at depth. Given by order, not name.
LAYER_PALETTE = (
    "#C9B896",  # sand
    "#B8A882",  # tan
    "#A89A76",  # khaki
    "#9AA38C",  # sage
    "#8B9A8B",  # green-grey
    "#7E9099",  # blue-grey
    "#6E8391",  # slate
    "#5F7585",  # deep slate
    "#556A7A",  # dark slate
    "#4A5D6B",  # charcoal blue
    "#414F5C",  # charcoal
    "#38434E",  # near-black
)

# The repository layer has this name in every case
HOST_ROCK = "Host_rock"


def _entry_value(entry):
    """Nominal depth of an interface entry (a list of one dict, or a dict)."""
    if isinstance(entry, list):
        if not entry:
            raise ValueError("Empty interface entry.")
        entry = entry[0]
    return float(entry["value"])


def read_interfaces(case_dir):
    """{interface_name: depth} from rock_interface.yaml."""
    path = Path(case_dir) / "geometry" / INTERFACE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Geometry file not found: {path}\n"
            f"Expected {INTERFACE_FILE} under {Path(case_dir) / 'geometry'}."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {name: _entry_value(entry) for name, entry in raw.items()}


def read_layers(case_dir, interfaces=None):
    """Layers as [{name, top, bottom, color}], from the surface down."""
    interfaces = interfaces or read_interfaces(case_dir)

    # every *_bottom is a layer; its top is the bottom above (or its own *_top)
    bottoms = {n[: -len("_bottom")]: z
               for n, z in interfaces.items() if n.endswith("_bottom")}
    tops = {n[: -len("_top")]: z
            for n, z in interfaces.items() if n.endswith("_top")}

    # z is negative downward, so descending z = surface to deep
    ordered = sorted(bottoms.items(), key=lambda kv: kv[1], reverse=True)

    layers = []
    previous_bottom = max(tops.values()) if tops else (
        max(bottoms.values()) if bottoms else 0.0
    )
    for i, (name, bottom) in enumerate(ordered):
        top = tops.get(name, previous_bottom)
        layers.append({
            "name": name,
            "top": top,
            "bottom": bottom,
            "color": LAYER_PALETTE[i % len(LAYER_PALETTE)],
        })
        previous_bottom = bottom

    return layers



def resolve_depth(name, case_dir=None, interfaces=None, layers=None):
    """Depth of an interface name like 'Host_rock_bottom'. Numbers are
    returned as they are. '<Layer>_top' is taken from the layer table when
    the file does not list it.
    """
    try:
        return float(name)
    except (TypeError, ValueError):
        pass

    interfaces = interfaces if interfaces is not None else read_interfaces(case_dir)
    if name in interfaces:
        return interfaces[name]

    if str(name).endswith("_top"):
        layer_name = str(name)[: -len("_top")]
        layers = layers if layers is not None else read_layers(case_dir, interfaces)
        for layer in layers:
            if layer["name"] == layer_name:
                return layer["top"]

    raise KeyError(
        f"Unknown depth '{name}'. Available interfaces: {sorted(interfaces)}\n"
        f"A layer's top can also be named '<Layer>_top' even when only "
        f"'<Layer>_bottom' is stored."
    )



def describe(case_dir):
    """Print the layers of a case."""
    layers = read_layers(case_dir)
    print(f"Stratigraphy for {case_dir}  ({len(layers)} layers)")
    for layer in layers:
        marker = "  <-- host rock" if layer["name"] == HOST_ROCK else ""
        print(f"  {layer['name']:<28} {layer['top']:>10.2f} .. "
              f"{layer['bottom']:>10.2f} m  {layer['color']}{marker}")
