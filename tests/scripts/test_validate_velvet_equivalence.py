# std-lib imports
import importlib.util
from pathlib import Path

# 3 party imports

# package imports


def load_validator():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "validate_velvet_equivalence.py"

    spec = importlib.util.spec_from_file_location(
        "validate_velvet_equivalence",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_velvet_validator_trajectoryflow_checks():
    validator = load_validator()

    assert validator.run_trajectoryflow_checks(
        verbose=False
    )
