"""
Trains a surrogate on the model box's output, validates it right away, and
saves it together with its validation metrics.

    python train_surrogate.py --config config.yaml --case_name Case3_Claystone \
        --uncertain_parameters uncertain_parameters.yaml \
        --training_data train.h5 --validation_data val.h5 \
        --case_folder . --engine_folder surrogate_models/engine_pce_3f9a1c2e

config.yaml is the surrogate block of the scenario yaml (with the defaults
applied). All locations are passed in; nothing is looked up by folder layout.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.experiment import Experiment
from src.runner import train_surrogate


def parse_args():
    parser = argparse.ArgumentParser(description="Train and validate a surrogate.")
    parser.add_argument("--config", required=True, help="yaml with the surrogate settings.")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--uncertain_parameters", required=True,
                        help="yaml with the uncertain_parameters block.")
    parser.add_argument("--training_data", required=True, help="Training model results (.h5).")
    parser.add_argument("--validation_data", default=None, help="Validation model results (.h5).")
    parser.add_argument("--case_folder", required=True,
                        help="Folder holding rock_data/, geometry/ and nuclide_water_diffusivity_data/.")
    parser.add_argument("--engine_folder", required=True,
                        help="Folder the engine, manifest and metrics are written to.")
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        surrogate_cfg = yaml.safe_load(f)

    exp = Experiment(
        name=args.case_name,
        path=Path(args.config),
        raw={
            "case": args.case_name,
            "training_data": str(Path(args.training_data).resolve()),
            "validation_data": (str(Path(args.validation_data).resolve())
                                if args.validation_data else None),
            "surrogate": surrogate_cfg,
        },
        parameters_file=Path(args.uncertain_parameters),
        case_folder=Path(args.case_folder),
        engine_folder=Path(args.engine_folder),
    )
    print(exp.describe())
    exp.require_training_data()

    engine = train_surrogate(
        meta_model_opts=exp.build_meta_model_opts(),
        exp_design_opts=None,
        n_init=exp.n_init,
        parameters_path=exp.parameters_path,
        case_dir=exp.case_dir,
        config_path=exp.config_path,
        training_path=exp.training_path,
        validation_path=exp.validation_path,
        output_qoi=exp.output_qoi,
        z_interest=exp.z_interest,
        t_interest=exp.t_interest,
        parallel=exp.surrogate.get("parallel", False),
        transform=exp.surrogate.get("transform"),
        qoi_floor=exp.qoi_floor,
        z_range=exp.z_range,
        z_min_variance=exp.z_min_variance,
        engine_path=exp.engine_path,
        verbose=True,
    )

    # Only now that engine.pkl exists do we write its manifest
    exp.write_manifest(exp.engine_path, stage="surrogate", extra={
        "kind": "engine",
        "case": exp.case,
        "engine_name": exp.engine_path.parent.name,
        "run_name": exp.run_name,
        "output_qoi": exp.output_qoi,
        "type": exp.surrogate_type,
        "specs": exp.surrogate_specs(),
        "n_nodes": (exp.n_nodes if exp.n_nodes is not None else len(engine.out_names)),
        "z_resolved": list(engine.out_names),
        "qoi_floor": exp.qoi_floor,
        "z_interest": exp.z_interest,
        "t_interest": exp.t_interest,
        "n_init": exp.n_init,
        # file names only; both files live in <hash8>/model_results/
        "trained_on": exp.training_path.name,
        "validated_on": (exp.validation_path.name
                         if exp.validation_path is not None else None),
    })
    print(f"\nSurrogate saved -> {exp.engine_path}")


if __name__ == "__main__":
    main()
