# Model card: credit default scoring (Home Credit)

Models: Logistic Regression, LightGBM, XGBoost, CatBoost, trained by `src/models/train.py` and registered in the MLflow Model Registry as `credit_scoring_{model}` (and `credit_scoring_{model}_base` for the single-table baseline). All numbers marked `<после dvc repro>` have not been computed yet: the Kaggle data is not available on the machine where this documentation was written, so the full pipeline has not been run end to end after the leakage fixes. They are filled in from MLflow after `uv run dvc repro`.

## Intended use

- **Purpose:** estimate the probability that a borrower will have payment difficulties (`TARGET = 1`), as the research and training engine behind the [LoanSight](https://github.com/mikhailvokhrameev/loan_sight.git) demo application.
- **Intended users:** the project owner and course reviewers experimenting with credit-risk modelling and MLOps practices.
- **Out of scope:** automated real-world lending decisions, any use on populations or periods other than the Home Credit sample, any use without human review and a compliance process. The model is not validated for production.

## Data

See [datasheet.md](datasheet.md). 85% of `application_train` is used for HPO, cross-validation and the final fit; 15% (`data.holdout_size`, split by `SK_ID_CURR`) is a holdout that no fit ever sees (encoders, feature selection, HPO, early stopping and thresholds are all fitted on the training part only). The Kaggle test set has no labels and is used only for drift (PSI).

## Model and training

- Features: aggregation of seven tables, top `features.top_n` (300) selected by Ridge on the training part.
- Hyperparameters: Optuna (`hpo.trials`, `hpo.n_splits`), seeds and costs from `params.yaml`; the Optuna study name embeds a hash of data and config.
- Final fit: 5-fold stratified CV gives out-of-fold (OOF) predictions for the OOF metrics and the threshold; the final model is refit on the whole training part and scored once on the holdout.
- Environment: `uv.lock`; pipeline: `dvc.yaml`.

## Metrics

Primary evaluation is on the holdout, computed from the MLflow run `Final_Production` after `uv run dvc repro` (`metrics.json` in `artifacts/<model>_model/`).

| Model | Holdout ROC-AUC | Holdout PR-AUC | Holdout Brier | Holdout business loss |
|---|---|---|---|---|
| LogReg | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` |
| LightGBM | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` |
| XGBoost | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` |
| CatBoost | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` | `<после dvc repro>` |

The README still shows the earlier results (ROC-AUC 0.751 to 0.788 across its tables). They are OOF numbers obtained before the leakage fix (F1), so they are optimistic; a holdout ROC-AUC lower than those OOF figures is expected after the fix.

## Decision threshold and cost

The threshold is the one that minimises the business loss `fn_cost * FN + fp_cost * FP` on the OOF predictions, with `business.fn_cost / business.fp_cost = 10 / 1`. A missed defaulter (approving a loan that is not repaid) is assumed to cost ten times a wrongly rejected good client. This ratio is an assumption for the exercise, not an estimate from Home Credit's economics; a real deployment needs the lender's actual loss given default and margin. The threshold is chosen on training OOF only and then applied unchanged to the holdout (`holdout_business_loss`). A client is approved when the predicted probability is below the threshold.

## Guardrails (logged on the holdout)

- **Calibration:** Brier score, reliability table and plot. Calibration is only monitored; no recalibration step (for example `CalibratedClassifierCV`) is applied.
- **Fairness:** approval-rate parity by `CODE_GENDER` and by age bin (`<30`, `30-39`, `40-49`, `50-59`, `60+`) at the business threshold. Reported per group: approval rate, approval rate among good borrowers, AUC. Summary: disparate impact (DI) ratio (min approval rate / max approval rate) and equal-opportunity (EO) gap. Groups with fewer than `fairness.min_group_size` (30) rows are shown but excluded from DI and EO. `holdout_fairness_min_di` is the minimum DI ratio over attributes.
- **Drift:** PSI of the score (OOF scores versus scores on the Kaggle test features) and of the top `monitoring.psi_top_k` (20) features. The run tag `drift_alert` is set to `true` when `psi_score` exceeds `monitoring.psi_alert` (0.25). Drift only warns; it never fails training.

## Fairness decisions

- `CODE_GENDER` is excluded from the model features (`features.exclude`) but kept in the holdout frame so the audit can still measure outcomes by sex. In the holdout frame the column is label-encoded (the encoder is fitted on the training part), so the fairness CSVs (`guardrails/fairness_gender.csv`) list groups as integer codes (for example 0/1/2), not as F/M/XNA; the encoders are not saved, but the mapping is deterministic: `LabelEncoder` sorts the training categories, so codes follow sorted category order (for example `F`=0, `M`=1, `__MISSING__`=2 when the training part has missing values) and `UNKNOWN`, used for categories unseen in the training part, gets the next code after the last one. `XNA` is turned into a missing value in the application tables and gets its own class (`__MISSING__`); a missing value in a part without that class falls under `UNKNOWN`. Dropping the column does not remove proxy effects, which the audit is meant to catch.
- Age (`DAYS_BIRTH`) stays a feature: it carries a strong monotone relationship with default (see Hypothesis 1 in the README) and is a standard underwriting variable, but it is a protected attribute in many jurisdictions. Its use should be reviewed against local regulation before any real deployment; the age-bin audit makes the effect visible.
- `NAME_FAMILY_STATUS` is not excluded and not audited separately (known gap).

## Promotion gate

`uv run python -m src.models.promote --model <name> [--base]` moves the `champion` alias to the `challenger` version only if

1. holdout ROC-AUC is at least champion holdout ROC-AUC minus `gate.auc_epsilon` (0.005), and
2. `holdout_fairness_min_di` is at least `fairness.di_min` (0.8).

Non-finite metrics fail closed (no promotion); the command exits with code 1 on rejection. The first model (no champion yet) is judged on the fairness criterion alone.

## Error budget and rollback

The champion is rolled back to the previous champion version if, on a fresh labelled cohort, ROC-AUC is below champion minus 0.02, or the score PSI exceeds 0.25, for two periods in a row. This follows the Zillow Offers lesson: a model that is not continuously checked against fresh outcomes can silently lose its edge, and the loss compounds before anyone notices. Note that the current code computes PSI against the Kaggle test sample only. Evaluation on a fresh labelled cohort, and the monitoring that triggers this rollback, are not implemented here and would have to be provided by the deployment.

## Limitations

- Labels exist only for granted loans (selection bias); the score is not validated on rejected applicants.
- One lender, one historical window: no evidence of transfer across time, geography or product.
- Holdout size is 15% of the labelled sample (about 46k rows with roughly 8% positives), so AUC differences of a few thousandths between models are within noise; do not rank close models on the holdout alone.
- Sensitive attributes beyond sex and age (for example family status) are not audited; proxy variables may still encode them.
- Probability calibration is monitored, not corrected; the score should not be read as a calibrated PD without checking the reliability table.
- SHAP: for XGBoost 3.x boosters the `shap` library cannot parse the model, so native TreeSHAP (`pred_contribs`) is used instead.
