# credit-scoring-mlops
Training/research pipeline for credit default scoring on Home Credit (7 relational tables): Python 3.10, pandas, LogReg/LightGBM/XGBoost/CatBoost, Optuna, MLflow, DVC. Exported artifacts are served by the separate LoanSight web app.

**IMPORTANT: Nothing is ever fit on the holdout (encoders, feature selection, HPO, early stopping, thresholds). It is scored exactly once, in `src/models/train.py`; ids in `data/processed/holdout_ids.json`.**
**YOU MUST NOT add `Co-Authored-By: Claude` or "Generated with Claude Code" to commits or PR descriptions.**

## Commands (run from repo root; env is uv, never conda/pip)
- Setup: `uv sync` (+ `--group notebooks` for Jupyter) · add dep: `uv add <pkg>`, never edit `uv.lock` by hand
- Pipeline graph: `uv run dvc dag` (stages: prepare → split → features → train@{model}; `features_base` is a separate branch with no train stage)
- Train: `uv run python -m src.models.train --model {logreg|lgbm|xgb|catboost} --trials 30 [--device gpu] [--base]`
- `--base` training needs the base features first: `uv run python -m src.features.pipeline_base`
- Promote to champion: `uv run python -m src.models.promote --model <name> [--base]` (exit 1 = rejected by gate)
- Lint + tests: `make check` · tests only: `uv run pytest -q`
- MLflow UI: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Anti-patterns
- Hardcoded params/paths/seeds/costs: everything comes from `params.yaml` via `src/config.load_params()`.
- Filling missing inference features with 0 or swallowing exceptions (e.g. turning failures into pruned trials): fail loudly.
- New copy-paste training scripts: extend `train.py` via flags.
- Tests that only raise coverage: test leakage, schema, reproducibility, metric math.
- README claims not backed by code or an MLflow run.
- Bloated PR descriptions; commits are small and imperative (`Add holdout split before selection`).

## Caveats
- `data/` is never in git; raw CSVs need a Kaggle token (`kaggle competitions download -c home-credit-default-risk`), then `data/raw` must be `dvc add`-ed after downloading (`uv run dvc add data/raw`).
- `test_features.parquet` = Kaggle test without labels: drift (PSI) only, never evaluation.
- Optuna study name embeds data+config hash; reusing a study mixes trials from other feature versions.
- Model wrappers must not mutate `self.params`: CV fold clones share it.
- `CODE_GENDER` is excluded from model features (`features.exclude`) but kept for the fairness audit; `DAYS_BIRTH` stays a feature. `promote.py` blocks `champion` when holdout `holdout_fairness_min_di` < `fairness.di_min`.
- `365243` in `DAYS_*` columns is a missing-value sentinel, handled in `src/data/preprocessor.py`.
- Notebooks only read DVC outputs/`artifacts/`; `make notebooks` executes 04 (needs data + the `lgbm` model and at least one other trained model); `tests/test_notebooks.py` checks `src` imports and cleared outputs.

## Docs (read on demand)
- Data: `docs/datasheet.md` · Model: `docs/model_card.md` · Params: `params.yaml`
