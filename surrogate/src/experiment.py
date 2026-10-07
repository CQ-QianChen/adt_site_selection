"""
experiment.py - One surrogate experiment: the surrogate settings plus the
locations of the data and of the engine to write.

All options are derived here. Locations are passed in (parameters_file,
case_folder, engine_folder); see train_surrogate.py.

    exp = Experiment(name=..., path=..., raw={...}, engine_folder=...)
    exp.training_path
    exp.engine_path

BayesValidRox is only imported where it is needed.
"""
from __future__ import annotations

import json
import re
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import warnings
import yaml

# repo root
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

# manifest format and registry names are shared with decision_support_uq
from .manifest import (  # noqa: F401  (re-exported)
    ENGINE_FILENAME,
    manifest_path_for,
    read_manifest,
)

def _resolve_surrogate_type(raw_type: str) -> str:
    """Normalize a type string to 'pce', 'gpe', 'gpe_torch' or 'pce_gpe'."""
    if raw_type is None:
        return "pce"
    s = str(raw_type).strip().lower()
    if "torch" in s:
        return "gpe_torch"
    if ("pc" in s and "gp" in s) or s in ("pce_gpe", "pce_gpr", "pcegp", "pcegpr"):
        return "pce_gpe"
    if "gp" in s or "gpe" in s or "gpr" in s:
        return "gpe"
    if "pc" in s or "pce" in s or "apce" in s:
        return "pce"
    raise ValueError(
        f"Unknown surrogate type: '{raw_type}'. Supported types are: "
        f"'pce', 'gpe', 'gpe_torch', 'pce_gpe'."
    )


def _import_bvr():
    """Import bayesvalidrox from the local submodule or the installed package."""
    try:
        from . import bayesvalidrox as bvr
        return bvr
    except (ImportError, ModuleNotFoundError):
        import bayesvalidrox as bvr
        return bvr


def _check_gpe_torch_available() -> bool:
    """Check if GPEGPy / GPETorch class is available in bayesvalidrox."""
    try:
        bvr = _import_bvr()
        return hasattr(bvr, "GPEGPy") or hasattr(bvr, "GPETorch")
    except Exception:
        return False


def _depth_sort_key(v):
    """Sort key for depths that mix floats and layer names."""
    try:
        return (0, float(v), "")
    except (ValueError, TypeError):
        return (1, 0.0, str(v))


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def _git_sha(cwd):
    """Short git SHA of the repo at `cwd`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _hash_files(paths):
    """Short hash of the content of some files."""
    h = hashlib.sha256()
    for p in sorted(str(Path(x)) for x in paths):
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:12]


# Data filenames look like ensemble_results_with_sample_{method}_size{N}_seed{S}.h5;
# the sampling design is read back from the name.
_SAMPLE_RE = re.compile(r"sample_(.+?)_size(\d+)_seed(\d+)")









# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    """One experiment, loaded from a yaml file."""
    name: str
    path: Path
    raw: dict
    workflow_dir: Path = WORKFLOW_DIR
    # Optional explicit locations; when unset the old workflow_dir layout is used.
    parameters_file: Path | None = None    # yaml with uncertain_parameters
    case_folder: Path | None = None        # has rock_data/, geometry/, nuclide_water_diffusivity_data/
    engine_folder: Path | None = None      # engine.hkl is written straight into this folder

    # ---- loading ---------------------------------------------------------- #
    def _req(self, key):
        if key not in self.raw:
            raise KeyError(f"Experiment '{self.name}' is missing required key: '{key}'")
        return self.raw[key]

    # ---- top-level settings ---------------------------------------------- #
    @property
    def case(self):
        """The case name, e.g. 'Case1_Claystone_Lower-Saxony'."""
        return str(self._req("case"))

    @property
    def node_tag(self):
        """Optional suffix for the output folder: '12_nodes' -> {Case}_12_nodes."""
        return self.raw.get("node_tag")

    # ---- input files ------------------------------------------------------ #
    @property
    def parameters_path(self):
        """Sampling config: which parameters are uncertain."""
        if self.parameters_file is not None:
            return Path(self.parameters_file)
        return (self.workflow_dir / "sampling" / "input"
                / f"sampling_config_{self.case}.yaml")

    @property
    def config_path(self):
        """Model config. Not needed for training, only recorded."""
        return self.workflow_dir / "model" / "input" / f"config_{self.case}.yaml"

    @property
    def has_sampled_geometry(self):
        """True if the sampling config has 'geometry' as uncertain (Case2).
        Then z_interest stays as names and data_io reads per sample.
        """
        if not self.parameters_path.exists():
            return False
        with open(self.parameters_path, "r", encoding="utf-8") as f:
            uncertain_parameters = yaml.safe_load(f) or {}
        block = uncertain_parameters.get("uncertain_parameters", uncertain_parameters)
        return bool(block.get("geometry"))

    @property
    def case_dir(self):
        """output/{Case}/"""
        if self.case_folder is not None:
            return Path(self.case_folder)
        return self.workflow_dir / "output" / self.case


    @property
    def surrogate_setup_dir(self):
        suffix = f"_{self.node_tag}" if self.node_tag else ""
        return self.workflow_dir / "output" / f"{self.case}{suffix}"

    @property
    def model_results_dir(self):
        return self.case_dir / "model_results"

    @property
    def training_path(self):
        """Training .h5 in model_results/."""
        val = self.raw.get("training_data") or self.raw.get("training_path")
        if val is not None:
            p = Path(val)
            return p if p.is_absolute() else self.model_results_dir / p.name
        raise KeyError(f"Experiment '{self.name}' requires 'training_data' in config")

    @property
    def validation_path(self):
        """Validation .h5 in model_results/, or None."""
        val = self.raw.get("validation_data") or self.raw.get("validation_path")
        if val is not None:
            p = Path(val)
            return p if p.is_absolute() else self.model_results_dir / p.name
        return None

    # ---- surrogate -------------------------------------------------------- #
    @property
    def surrogate(self):
        return self.raw.get("surrogate") or {}

    @property
    def output_qoi(self):
        return self.surrogate.get("output_qoi", "I-129")

    def _normalize_interval(self, val1, val2):
        """Normalize an interval [val1, val2] to (top, bottom) with top >= bottom."""
        if val1 is None and val2 is None:
            return (None, None)

        if val1 is not None and not self.has_sampled_geometry:
            from .geometry import resolve_depth
            try:
                val1 = float(resolve_depth(val1, self.case_dir))
            except Exception:
                try:
                    val1 = float(val1)
                except (ValueError, TypeError):
                    pass
        elif val1 is not None:
            try:
                val1 = float(val1)
            except (ValueError, TypeError):
                pass

        if val2 is not None and not self.has_sampled_geometry:
            from .geometry import resolve_depth
            try:
                val2 = float(resolve_depth(val2, self.case_dir))
            except Exception:
                try:
                    val2 = float(val2)
                except (ValueError, TypeError):
                    pass
        elif val2 is not None:
            try:
                val2 = float(val2)
            except (ValueError, TypeError):
                pass

        if val1 is not None and val2 is not None:
            try:
                f1, f2 = float(val1), float(val2)
                return (max(f1, f2), min(f1, f2))
            except (ValueError, TypeError):
                return (val1, val2)

        return (val1, val2)

    def _parse_z_config(self):
        """Parse surrogate.z_interest into (discrete_depths_or_names, list_of_ranges)."""
        z = self.surrogate.get("z_interest")
        if z is None:
            raise KeyError(f"Experiment '{self.name}' requires surrogate.z_interest")

        if isinstance(z, str) and z.strip().lower() in ("all", "whole_column"):
            return [], [(None, None)]

        if isinstance(z, (int, float)):
            return [float(z)], []

        if isinstance(z, dict):
            r = z.get("range")
            if r is None:
                raise KeyError(
                    f"Experiment '{self.name}': mapping z_interest requires 'range: [top, bottom]'"
                )
            top, bottom = (list(r) + [None, None])[:2]
            return [], [self._normalize_interval(top, bottom)]

        if isinstance(z, str):
            z = [z]

        discrete = []
        ranges = []
        interfaces = None

        for item in z:
            if item is None:
                continue

            if isinstance(item, dict):
                r = item.get("range") or item.get("interval")
                if r is not None:
                    top, bottom = (list(r) + [None, None])[:2]
                    ranges.append(self._normalize_interval(top, bottom))
                continue

            if isinstance(item, (list, tuple)):
                top, bottom = (list(item) + [None, None])[:2]
                ranges.append(self._normalize_interval(top, bottom))
                continue

            if isinstance(item, str) and item.strip().lower() in ("all", "whole_column"):
                ranges.append((None, None))
                continue

            try:
                discrete.append(float(item))
                continue
            except (ValueError, TypeError):
                pass

            if self.has_sampled_geometry:
                discrete.append(str(item))
                continue

            if interfaces is None:
                from .geometry import read_interfaces
                interfaces = read_interfaces(self.case_dir)
            from .geometry import resolve_depth
            try:
                discrete.append(float(resolve_depth(item, self.case_dir, interfaces)))
            except KeyError as exc:
                raise KeyError(
                    f"z_interest '{item}' is not a boundary of case '{self.case}'. {exc}"
                ) from None

        return discrete, ranges

    @property
    def z_discrete(self):
        """List of discrete depths or boundary names requested."""
        discrete, _ = self._parse_z_config()
        return discrete

    @property
    def z_range(self):
        """Depth interval(s) to train on: list of (top, bottom) tuples, or None."""
        _, ranges = self._parse_z_config()
        return ranges if ranges else None

    @property
    def z_interest(self):
        """Depths to train at: numbers or boundary names.
        Returns list of discrete depths/names, or None if only range(s) were specified.
        """
        discrete, ranges = self._parse_z_config()
        return discrete if discrete else None

    @property
    def z_min_variance(self):
        """Drop depths that vary less than this across samples. Default: 1e-30."""
        v = self.surrogate.get("z_min_variance")
        if v is None:
            v = self.surrogate.get("min_variance")
        if v is None and isinstance(self.surrogate.get("z_interest"), dict):
            v = self.surrogate["z_interest"].get("min_variance")
        if v is None:
            v = 1e-30
        return float(v)

    @property
    def qoi_floor(self):
        """Numerical noise floor threshold. Can be a float or dict mapping QoI -> float.
        If omitted from config, defaults to 1e-20 for flux QoIs, and 1e-12 for concentration QoIs.
        """
        val = self.surrogate.get("qoi_floor")
        is_flux = "flux" in self.output_qoi.lower()
        default_val = 1e-20 if is_flux else 1e-12

        if val is None:
            return default_val
        if isinstance(val, dict):
            qoi_val = val.get(self.output_qoi)
            return float(qoi_val) if qoi_val is not None else default_val
        return float(val)

    @property
    def transform(self):
        """Output transform for training, or None."""
        t = self.surrogate.get("transform")
        if t in (None, "", "none"):
            return None
        return str(t)

    @property
    def t_interest(self):
        """Time steps to train at, or None for all. A single value is allowed."""
        t = self.surrogate.get("t_interest")
        if t is None:
            return None
        if isinstance(t, (int, float, str)):
            t = [t]
        return [float(v) for v in t]

    @property
    def n_init(self):
        """Number of training points to use; None = all in the file."""
        n = self.surrogate.get("n_init")
        return int(n) if n is not None else None

    @property
    def surrogate_type(self):
        """One of 'pce', 'gpe', 'gpe_torch', 'pce_gpe'."""
        stype = _resolve_surrogate_type(self.surrogate.get("type", "pce"))
        if stype == "gpe_torch":
            if not _check_gpe_torch_available():
                warnings.warn(
                    "surrogate.type: 'gpe_torch' (GPETorch) is not available in this "
                    "bayesvalidrox version; falling back to 'gpe' (GPESkl).",
                    UserWarning,
                    stacklevel=2,
                )
                return "gpe"
        return stype

    @property
    def pce_options(self):
        """PCE options from the yaml, over the defaults."""
        pce_dict = (
            self.surrogate.get("pce_opts")
            or self.surrogate.get("pce")
            or self.surrogate.get("pc")
            or self.surrogate.get("apce")
            or {}
        )
        if not isinstance(pce_dict, dict):
            pce_dict = {}

        defaults = {
            "type": "aPCE",
            "reg_method": "OLS",
            "q_norm": 1.0,
            "degree": [1],
            "dim_red_method": "no",
            "bootstrap_method": "fast",
            "n_bootstrap_itrs": 1,
            "input_transform": "user",
        }
        res = {**defaults, **pce_dict}
        if res.get("dim_red_method") in (False, None):
            res["dim_red_method"] = "no"
        return res

    @property
    def gp_options(self):
        """GP options from the yaml, over the defaults."""
        gp_dict = (
            self.surrogate.get("gp_opts")
            or self.surrogate.get("gp")
            or self.surrogate.get("gpe")
            or self.surrogate.get("gpr")
            or {}
        )
        if not isinstance(gp_dict, dict):
            gp_dict = {}

        defaults = {
            "kernel_type": "RBF",
            "isotropy": False,
            "auto_select": False,
            "n_restarts": 10,
            "noisy": False,
            "nugget": 1.0e-9,
            "normalize_x_method": "norm",
            "transform_y_method": "norm",
            "gpe_reg_method": "LBFGS",
            "dim_red_method": "no",
            "input_transform": "user",
        }
        if "kernel" in gp_dict and "kernel_type" not in gp_dict:
            gp_dict["kernel_type"] = gp_dict["kernel"]

        res = {**defaults, **gp_dict}
        if res.get("dim_red_method") in (False, None):
            res["dim_red_method"] = "no"
        return res

    @property
    def pce_gpe_options(self):
        """PCEGPR options, from pce_opts and gp_opts."""
        pce = self.pce_options
        gp = self.gp_options
        deg = pce.get("degree", [1])
        deg_scalar = max(deg) if isinstance(deg, (list, tuple)) else int(deg)

        return {
            "meta_model_type": "PCE_GPR",
            "pce_model_type": pce.get("type", "aPCE"),
            "pce_reg_method": pce.get("reg_method", "OLS"),
            "pce_deg": deg_scalar,
            "pce_q_norm": float(pce.get("q_norm", 1.0)),
            "kernel_type": gp.get("kernel_type", "RBF"),
            "auto_select": gp.get("auto_select", False),
            "noisy": True,  # PCEGPR is always noisy
            "normalize_x_method": gp.get("normalize_x_method", "norm"),
            "dim_red_method": pce.get("dim_red_method", "no"),
            "bootstrap_method": pce.get("bootstrap_method", "fast"),
            "n_bootstrap_itrs": pce.get("n_bootstrap_itrs", 1),
            "input_transform": pce.get("input_transform", "user"),
        }

    @property
    def _q_tag(self):
        """Hyperparameter suffix for the run name."""
        st = self.surrogate_type
        if st in ("gpe", "gpe_torch"):
            gp = self.gp_options
            iso = "iso" if gp["isotropy"] else "aniso"
            tag = f"_{gp['kernel_type']}{iso}r{int(gp['n_restarts'])}"
            if gp["auto_select"]:
                tag += "auto"
            if gp["noisy"]:
                tag += "noisy"
            return tag

        if st == "pce_gpe":
            pg = self.pce_gpe_options
            q_str = str(pg["pce_q_norm"]).replace(".", "")
            tag = f"_q{q_str}d{pg['pce_deg']}_{pg['kernel_type']}"
            if pg["auto_select"]:
                tag += "auto"
            return tag

        pce = self.pce_options
        deg = pce["degree"]
        deg_str = "-".join(str(d) for d in deg) if isinstance(deg, (list, tuple)) else str(int(deg))
        q_str = str(pce["q_norm"]).replace(".", "")
        return f"_q{q_str}d{deg_str}"

    @property
    def _transform_tag(self):
        """Transform suffix for the run name, if any."""
        return f"_x{self.transform}" if self.transform else ""

    @property
    def reg_method(self):
        if self.surrogate_type in ("gpe", "gpe_torch"):
            return ""
        if self.surrogate_type == "pce_gpe":
            return self.pce_gpe_options["pce_reg_method"]
        return self.pce_options["reg_method"]

    @property
    def q_norm(self):
        return self.pce_options["q_norm"]

    @property
    def degree(self):
        deg = self.pce_options["degree"]
        return [int(deg)] if isinstance(deg, (int, float)) else deg

    @property
    def n_nodes(self):
        """Number of depths, or None in range mode (known only after loading)."""
        z = self.z_interest
        return None if z is None else len(z)

    @property
    def run_name(self):
        """Name for this config: {type}_{reg}_{qoi}_{N}N_{n}TP_q{q}d{deg}{x}."""
        tp_tag = f"{self.n_init}TP" if self.n_init is not None else "allTP"
        n_tag = "range" if self.n_nodes is None else self.n_nodes
        base = (f"{self.surrogate_type}{'_' + self.reg_method if self.reg_method else ''}"
                f"_{self.output_qoi}_{n_tag}N_{tp_tag}")
        return base + self._q_tag + self._transform_tag

    def surrogate_specs(self):
        """Settings that define the engine (for the fingerprint and manifest)."""
        st = self.surrogate_type
        base = {
            "case": self.case,
            "node_tag": self.node_tag,
            "output_qoi": self.output_qoi,
            "type": st,
            "n_nodes": self.n_nodes,
            "z_interest": (sorted(self.z_interest, key=_depth_sort_key)
                           if self.z_interest is not None else None),
            "z_range": self.z_range,
            "z_min_variance": self.z_min_variance,
            "t_interest": self.t_interest,
            "transform": self.transform,
            "qoi_floor": self.qoi_floor,
            "n_init": self.n_init,
            "training_data": self.training_path.name,
        }
        if st in ("gpe", "gpe_torch"):
            return {**base, "gp_opts": self.gp_options}
        if st == "pce_gpe":
            return {**base, "pce_opts": self.pce_options, "gp_opts": self.gp_options}
        return {**base, "pce_opts": self.pce_options}

    # ---- engine registry (config -> engine_v{N}) -------------------------- #
    @property
    def engine_path(self):
        return Path(self.engine_folder) / ENGINE_FILENAME

    def build_meta_model_opts(self):
        """Function that builds the untrained surrogate from `inputs`.

          pce       -> PCE
          gpe       -> GPESkl  (scikit-learn GP)
          gpe_torch -> GPEGPy  (PyTorch GP)
          pce_gpe   -> PCEGPR
        """
        bvr = _import_bvr()
        stype = self.surrogate_type

        # ---- GP (PyTorch) ---------------------------------------------- #
        if stype == "gpe_torch":
            TorchGP = getattr(bvr, "GPEGPy", None) or getattr(bvr, "GPETorch", None)
            if TorchGP is None:
                raise ValueError(
                    "surrogate.type: 'gpe_torch' requested but neither 'GPEGPy' nor 'GPETorch' "
                    "was found in bayesvalidrox."
                )
            gp = self.gp_options
            return lambda inputs: TorchGP(
                input_obj=inputs,
                meta_model_type="GPE",
                kernel_type=gp.get("kernel_type", "RBF"),
                isotropy=gp.get("isotropy", False),
                auto_select=gp.get("auto_select", False),
                n_restarts=gp.get("n_restarts", 10),
                noisy=gp.get("noisy", False),
                nugget=gp.get("nugget", 1e-5),
                normalize_x_method=gp.get("normalize_x_method", "norm"),
                dim_red_method=gp.get("dim_red_method", "no"),
                input_transform=gp.get("input_transform", "user"),
            )

        # ---- GP (scikit-learn) ----------------------------------------- #
        if stype == "gpe":
            GPESkl = bvr.GPESkl
            gp = self.gp_options
            return lambda inputs: GPESkl(
                input_obj=inputs,
                meta_model_type="GPE",
                kernel_type=gp.get("kernel_type", "RBF"),
                isotropy=gp.get("isotropy", False),
                auto_select=gp.get("auto_select", False),
                n_restarts=gp.get("n_restarts", 10),
                noisy=gp.get("noisy", False),
                nugget=gp.get("nugget", 1e-9),
                normalize_x_method=gp.get("normalize_x_method", "norm"),
                transform_y_method=gp.get("transform_y_method", "norm"),
                gpe_reg_method=gp.get("gpe_reg_method", "LBFGS"),
                dim_red_method=gp.get("dim_red_method", "no"),
                input_transform=gp.get("input_transform", "user"),
            )

        # ---- PCE+GP hybrid ---------------------------------------- #
        if stype == "pce_gpe":
            PCEGPR = bvr.PCEGPR
            pg = self.pce_gpe_options
            return lambda inputs: PCEGPR(
                input_obj=inputs,
                meta_model_type=pg.get("meta_model_type", "PCE_GPR"),
                pce_model_type=pg.get("pce_model_type", "aPCE"),
                pce_reg_method=pg.get("pce_reg_method", "OLS"),
                pce_deg=pg.get("pce_deg", 1),
                pce_q_norm=pg.get("pce_q_norm", 1.0),
                kernel_type=pg.get("kernel_type", "RBF"),
                auto_select=pg.get("auto_select", False),
                normalize_x_method=pg.get("normalize_x_method", "norm"),
                dim_red_method=pg.get("dim_red_method", "no"),
                bootstrap_method=pg.get("bootstrap_method", "fast"),
                n_bootstrap_itrs=pg.get("n_bootstrap_itrs", 1),
                input_transform=pg.get("input_transform", "user"),
            )

        # ---- PCE -------------------------------------------------- #
        PCE = bvr.PCE
        pce = self.pce_options
        return lambda inputs: PCE(
            input_obj=inputs,
            meta_model_type=pce.get("type", "aPCE"),
            pce_reg_method=pce.get("reg_method", "OLS"),
            pce_q_norm=pce.get("q_norm", 1.0),
            pce_deg=pce.get("degree", [1]),
            dim_red_method=pce.get("dim_red_method", "no"),
            bootstrap_method=pce.get("bootstrap_method", "fast"),
            n_bootstrap_itrs=pce.get("n_bootstrap_itrs", 1),
            input_transform=pce.get("input_transform", "user"),
        )

    # ---- manifests ------------------------------------------ #
    def config_hash(self):
        """Hash of this yaml and the two configs it points to."""
        return _hash_files([self.path, self.parameters_path, self.config_path])

    def write_manifest(self, target_file, stage, extra=None):
        """Write a JSON manifest next to `target_file` and return its path."""
        payload = {
            "experiment": self.name,
            "stage": stage,
            "case": self.case,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config_hash": self.config_hash(),
            "git_sha": _git_sha(self.workflow_dir),
        }
        if extra:
            payload.update(extra)
        man = manifest_path_for(target_file)
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(payload, indent=2) + "\n")
        return man

    # ---- checks ---- #
    def require_training_data(self):
        """Stop early if the training file is missing or from another case."""
        if not self.training_path.exists():
            raise FileNotFoundError(
                f"Training data not found:\n  {self.training_path}\n\n"
                f"Run stage 1 first with this experiment "
                f"(EXPERIMENT = '{self.path.name}' in run_sampling.py)."
            )
        man = read_manifest(self.training_path)
        if man and man.get("case") not in (None, self.case):
            raise ValueError(
                f"Training data {self.training_path.name} was produced for case "
                f"'{man.get('case')}', but this experiment is '{self.case}'. "
                f"Re-run the sampling and model boxes, or fix the config."
            )

    # ---- summary ------------------------------------------------------------- #
    def describe(self):
        try:
            train_rel = self.training_path.relative_to(self.workflow_dir)
        except Exception:
            train_rel = self.training_path
        n_init_str = f" (n_init={self.n_init})" if self.n_init is not None else ""
        lines = [
            f"Experiment : {self.name}   ({self.path})",
            f"Case       : {self.case}",
            f"Train      : {self.training_path.name}{n_init_str}  -> {train_rel}",
        ]
        if self.validation_path is not None:
            try:
                val_rel = self.validation_path.relative_to(self.workflow_dir)
            except Exception:
                val_rel = self.validation_path
            lines.append(f"Validation : {self.validation_path.name}  -> {val_rel}")
        else:
            lines.append("Validation : (none)")
        try:
            engine_rel = self.engine_path.relative_to(self.workflow_dir)
        except Exception:
            engine_rel = self.engine_path
        status = "exists" if self.engine_path.exists() else "not trained yet"
        if self.z_discrete and self.z_range:
            depth_desc = f"discrete {self.z_discrete}, range {self.z_range} (min_var {self.z_min_variance:g})"
        elif self.z_range:
            depth_desc = f"range {self.z_range} (min_var {self.z_min_variance:g})"
        elif self.z_discrete:
            depth_desc = f"{self.z_discrete}"
        else:
            depth_desc = "(none)"
        lines += [
            f"Surrogate  : {self.run_name}",
            f"Type       : {self.surrogate_type}",
            f"QoI        : {self.output_qoi} (qoi_floor={self.qoi_floor:g})",
            f"Engine     -> {engine_rel}  ({status})",
            f"Depths     : {depth_desc}",
        ]
        return "\n".join(lines)
