"""Stratified holdout split by SK_ID_CURR, made before any feature selection or encoder fit."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_params
from src.data.loader import DataLoader

logger = logging.getLogger(__name__)

HOLDOUT_IDS_PATH = ROOT / "data" / "processed" / "holdout_ids.json"


def make_holdout_ids(ids: pd.Series, target: pd.Series, size: float, seed: int) -> np.ndarray:
    """Return the sorted SK_ID_CURR of a target-stratified holdout of fraction `size`."""
    _, holdout = train_test_split(
        np.asarray(ids), test_size=size, stratify=np.asarray(target), random_state=seed
    )
    return np.sort(holdout)


def save_ids(ids, path: Path) -> None:
    """Write ids as a JSON list of ints (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([int(i) for i in ids]))


def load_ids(path: Path) -> np.ndarray:
    """Read ids written by `save_ids`."""
    return np.asarray(json.loads(Path(path).read_text()), dtype=np.int64)


def load_holdout_ids(path: Path = HOLDOUT_IDS_PATH) -> np.ndarray:
    """Read the holdout ids; FileNotFoundError tells how to create them."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: run `python -m src.data.split` first")
    return load_ids(path)


def main() -> None:
    params = load_params()
    loader = DataLoader(raw_dir=str(ROOT / "data" / "raw"), cache_dir=str(ROOT / "data" / "cache"))
    train = loader.load_table("application_train")
    holdout = make_holdout_ids(
        train["SK_ID_CURR"], train["TARGET"], params["data"]["holdout_size"], params["seed"]
    )
    save_ids(holdout, HOLDOUT_IDS_PATH)
    logger.info("Saved %d holdout ids to %s", len(holdout), HOLDOUT_IDS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    main()
