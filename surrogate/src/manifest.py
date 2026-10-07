"""
manifest.py - The JSON manifest written next to output files, and the engine
registry filenames.

The surrogate box writes one next to engine.pkl; the decision-support box reads
it to get the QoI, case and depths of an engine.
"""
import json
from pathlib import Path

_MANIFEST_SUFFIX = ".manifest.json"
ENGINE_FILENAME = "engine.pkl"


def manifest_path_for(target_file):
    """Path of the manifest next to an output file."""
    return Path(str(target_file) + _MANIFEST_SUFFIX)


def read_manifest(target_file):
    """Manifest of an output file as a dict, or None."""
    man = manifest_path_for(target_file)
    if man.exists():
        try:
            return json.loads(man.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None
