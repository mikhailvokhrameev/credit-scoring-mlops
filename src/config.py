"""Single source of truth for tunable constants (seeds, costs, paths, MLflow)."""
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SECTIONS = (
    "seed", "data", "features", "cv", "hpo", "train",
    "business", "mlflow", "monitoring", "fairness", "gate",
)


def load_params(path: Path | None = None) -> dict:
    """Load params.yaml (default: ROOT / "params.yaml"); KeyError names all missing sections."""
    path = Path(path) if path is not None else ROOT / "params.yaml"
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    missing = [s for s in REQUIRED_SECTIONS if s not in params]
    if missing:
        raise KeyError(f"Missing required section(s) {missing} in {path}")
    return params


def stable_hash(obj: Any) -> str:
    """Order-independent sha256 hex digest of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
