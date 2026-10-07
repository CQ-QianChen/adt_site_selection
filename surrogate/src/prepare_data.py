"""Input parameters and distributions for BayesValidRox."""

import numpy as np
from pathlib import Path
import scipy
import scipy.stats as st
import yaml

# Order of parameter groups shared across surrogate data handling (defined in data_io)
from .data_io import GROUP_ORDER


def parse_distribution_spec(spec: str):
    """Parse a probability distribution string from the data-hub.

    Strips any trailing `.rvs(...)` call and safely evaluates the scipy.stats
    expression to return the frozen distribution object.
    """
    if not isinstance(spec, str):
        raise TypeError(f"Expected string for distribution spec, got {type(spec)}")

    rvs_idx = spec.find(".rvs(")
    expr = spec[:rvs_idx].strip() if rvs_idx != -1 else spec.strip()

    # Restrict execution environment to scipy.stats distributions
    safe_env = {"scipy": scipy, "stats": st}
    for name in dir(st):
        attr = getattr(st, name)
        if hasattr(attr, "rvs") or hasattr(attr, "pdf") or hasattr(attr, "cdf"):
            safe_env[name] = attr

    try:
        return eval(expr, {"__builtins__": {}}, safe_env)
    except Exception as e:
        raise ValueError(f"Failed to parse distribution expression '{expr}': {e}") from e


def get_params_chaospy(spec: str):
    """Extract parameters and distribution name for BayesValidRox from a spec string.

    Supports the 8 distributions recognized by BayesValidRox InputSpace:
    'uniform', 'truncnorm', 'lognorm', 'norm', 'gamma', 'beta', 'weibull', and 'expon'.
    """
    dist_obj = parse_distribution_spec(spec)
    dist_name = dist_obj.dist.name

    if dist_name == "uniform":
        loc = dist_obj.kwds.get("loc", dist_obj.args[0] if len(dist_obj.args) > 0 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[1] if len(dist_obj.args) > 1 else 1.0)
        return [float(loc), float(loc + scale)], "uniform"

    elif dist_name == "truncnorm":
        a = dist_obj.kwds.get("a", dist_obj.args[0] if len(dist_obj.args) > 0 else None)
        b = dist_obj.kwds.get("b", dist_obj.args[1] if len(dist_obj.args) > 1 else None)
        loc = dist_obj.kwds.get("loc", dist_obj.args[2] if len(dist_obj.args) > 2 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[3] if len(dist_obj.args) > 3 else 1.0)
        lower_chaos = float((a * scale) + loc)
        upper_chaos = float((b * scale) + loc)
        return [lower_chaos, upper_chaos, float(loc), float(scale)], "truncnorm"

    elif dist_name == "lognorm":
        s = dist_obj.kwds.get("s", dist_obj.args[0] if len(dist_obj.args) > 0 else None)
        scale = dist_obj.kwds.get("scale", dist_obj.args[1] if len(dist_obj.args) > 1 else 1.0)
        loc = dist_obj.kwds.get("loc", dist_obj.args[2] if len(dist_obj.args) > 2 else 0.0)
        m = scale * np.exp((s**2) / 2)
        v = (scale**2 * np.exp(s**2)) * (np.exp(s**2) - 1)
        std_dev = np.sqrt(v)
        return [float(m), float(std_dev)], "lognorm"

    elif dist_name == "norm":
        loc = dist_obj.kwds.get("loc", dist_obj.args[0] if len(dist_obj.args) > 0 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[1] if len(dist_obj.args) > 1 else 1.0)
        return [float(loc), float(scale)], "norm"

    elif dist_name == "gamma":
        a = dist_obj.kwds.get("a", dist_obj.args[0] if len(dist_obj.args) > 0 else None)
        loc = dist_obj.kwds.get("loc", dist_obj.args[1] if len(dist_obj.args) > 1 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[2] if len(dist_obj.args) > 2 else 1.0)
        return [float(a), float(scale), float(loc)], "gamma"

    elif dist_name == "beta":
        a = dist_obj.kwds.get("a", dist_obj.args[0] if len(dist_obj.args) > 0 else None)
        b = dist_obj.kwds.get("b", dist_obj.args[1] if len(dist_obj.args) > 1 else None)
        loc = dist_obj.kwds.get("loc", dist_obj.args[2] if len(dist_obj.args) > 2 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[3] if len(dist_obj.args) > 3 else 1.0)
        return [float(a), float(b), float(loc), float(loc + scale)], "beta"

    elif dist_name in ("weibull", "weibull_min"):
        c = dist_obj.kwds.get("c", dist_obj.args[0] if len(dist_obj.args) > 0 else None)
        loc = dist_obj.kwds.get("loc", dist_obj.args[1] if len(dist_obj.args) > 1 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[2] if len(dist_obj.args) > 2 else 1.0)
        return [float(c), float(scale), float(loc)], "weibull"

    elif dist_name in ("expon", "exponential"):
        loc = dist_obj.kwds.get("loc", dist_obj.args[0] if len(dist_obj.args) > 0 else 0.0)
        scale = dist_obj.kwds.get("scale", dist_obj.args[1] if len(dist_obj.args) > 1 else 1.0)
        return [float(scale), float(loc)], "expon"

    else:
        raise ValueError(
            f"Distribution '{dist_name}' from spec '{spec[:80]}' is not supported "
            "by BayesValidRox. Supported distributions are: "
            "'uniform', 'truncnorm', 'lognorm', 'norm', 'gamma', 'beta', 'weibull', and 'expon'."
        )

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spec_from_entry(entry):
    """Distribution string of a data-hub entry (a list of one dict, or a dict)."""
    if isinstance(entry, list):
        if not entry:
            raise ValueError("Empty property entry in data file.")
        entry = entry[0]
    return entry["probability_distribution"]["sampled_data"]


def iter_uncertain(uncertain_parameters):
    """Yield (param_name, folder, layer, prop) for every uncertain parameter.

    folder: key in the config and folder under output/{Case}/
    layer:  yaml file in that folder (without .yaml)
    prop:   key inside that file
    geometry is different: geometry -> rock_interface -> [interface names].

    This order is the column order of X.
    """
    block = uncertain_parameters.get("uncertain_parameters", uncertain_parameters)

    ordered = [g for g in GROUP_ORDER if g in block]
    ordered += sorted(g for g in block if g not in GROUP_ORDER)

    for folder in ordered:
        layers = block[folder] or {}
        for layer in sorted(layers.keys()):
            props = layers[layer] or []
            for prop in props:
                yield f"{folder}_{layer}_{prop}", folder, layer, prop


def get_param_names_from_config(uncertain_parameters):
    """Parameter names, in the column order of X."""
    return [name for name, _, _, _ in iter_uncertain(uncertain_parameters)]


def extract_distributions(uncertain_parameters, case_dir):
    """Distribution of each uncertain parameter, read from the case folders.

    uncertain_parameters: the parsed sampling config.
    case_dir: output/{Case}/
    Returns [{name, dist_type, parameters}], in the same order as the names.
    """
    case_dir = Path(case_dir)
    distributions = []
    cache = {}

    for name, folder, layer, prop in iter_uncertain(uncertain_parameters):
        if folder == "geometry":
            # one file, keys are the interface names
            path = case_dir / folder / f"{layer}.yaml"
            key = prop
        else:
            # folder/{layer}.yaml, property is a key inside
            path = case_dir / folder / f"{layer}.yaml"
            key = prop

        if not path.exists():
            raise FileNotFoundError(
                f"Data file for '{name}' not found: {path}\n"
                f"The config key '{folder}' must match a folder under {case_dir}."
            )

        if path not in cache:
            cache[path] = _load_yaml(path)
        data = cache[path]

        if key not in data:
            raise KeyError(
                f"Property '{key}' not found in {path}. "
                f"Available: {sorted(data.keys())[:10]}"
            )

        spec = _spec_from_entry(data[key])
        params, dist_name = get_params_chaospy(spec)

        distributions.append({
            "name": name,
            "dist_type": dist_name,
            "parameters": params,
        })

    return distributions



def get_output_names(z_interest=None, t_interest=None, output_qoi="I-129"):
    """Output names for the BVR model from z_interest."""
    if z_interest is not None:
        z_list = z_interest if isinstance(z_interest, (list, np.ndarray)) else [z_interest]
        # numbers -> '-225.0'; names stay as they are
        def _name(z):
            try:
                return str(float(z))
            except (ValueError, TypeError):
                return str(z)
        return [_name(z) for z in z_list]
    elif t_interest is not None:
        if isinstance(t_interest, (int, float, np.number)):
            return [f"t_{float(t_interest)}"]
        else:
            return [f"t_{float(t)}" for t in t_interest]
    else:
        return [output_qoi]


def create_input_space(uncertain_parameters, case_dir):
    """BayesValidRox Input with the marginals read from the case folders."""
    from .bayesvalidrox import Input
    from .bayesvalidrox.surrogate_models.input_space import InputSpace

    inputs = Input()

    # distributions from the case folders
    distributions = extract_distributions(uncertain_parameters, case_dir)

    # add the marginals in order
    for dist in distributions:
        inputs.add_marginals(
            name=dist["name"],
            dist_type=dist["dist_type"],
            parameters=list(dist["parameters"]),
        )
        
    # build the input space
    input_space = InputSpace(inputs)
    input_space.init_param_space()
    
    return inputs