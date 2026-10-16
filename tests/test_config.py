import re

import pytest
import yaml

from src.config import ROOT, load_params, stable_hash


def test_load_default_params():
    p = load_params()
    assert p["seed"] == 42 and p["features"]["top_n"] == 300


def test_override_file(tmp_path):
    src = yaml.safe_load((ROOT / "params.yaml").read_text())
    src["features"]["top_n"] = 200
    src["business"]["fn_cost"] = 5.0
    f = tmp_path / "params.yaml"
    f.write_text(yaml.safe_dump(src))
    p = load_params(f)
    assert p["features"]["top_n"] == 200 and p["business"]["fn_cost"] == 5.0


def test_missing_section_names_key(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("seed: 1\n")
    with pytest.raises(KeyError, match="features"):
        load_params(f)


def test_stable_hash_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_no_hardcoded_seed_in_src():
    hits = [
        p for p in (ROOT / "src").rglob("*.py")
        if re.search(r"random_state\s*=\s*42|seed\s*=\s*42|fn_cost\s*=\s*10", p.read_text())
    ]
    assert hits == []
