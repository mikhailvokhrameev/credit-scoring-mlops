# Datasheet: Home Credit Default Risk

Documentation of the data used by this repository, following the "Datasheets for Datasets" structure. Parameters quoted below live in [`params.yaml`](../params.yaml).

## Provenance

- **Source:** Kaggle competition [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk), published by Home Credit Group (consumer lending to people with little or no credit history).
- **Obtained with:** `kaggle competitions download -c home-credit-default-risk` (requires a Kaggle API token and acceptance of the competition rules).
- **License / terms of use:** per the Kaggle competition page, the data is distributed under the competition rules, not an open data license (the exact wording was not re-checked here). Use is limited to what the competition rules allow (check the current rules before any use beyond this research project) and the raw files must not be redistributed. For that reason `data/` is never committed to git: the raw CSVs are meant to be tracked by DVC locally (`uv run dvc add data/raw`) and no DVC remote is configured.
- **Collection process:** historical loan applications of Home Credit clients and their credit-bureau and internal repayment histories. How the sample was drawn and anonymised is defined by the competition organisers and is not documented beyond the Kaggle data description.

## Composition

Seven relational tables, joined to the borrower via `SK_ID_CURR`:

| Table | Grain | Content |
|---|---|---|
| `application_train` / `application_test` | one row per loan application | Application form, `TARGET` (train only) |
| `bureau` | one row per previous credit at other institutions | Credit-bureau history |
| `bureau_balance` | one row per month per bureau credit | Monthly status of bureau credits |
| `previous_application` | one row per earlier Home Credit application | Earlier applications at Home Credit |
| `POS_CASH_balance` | one row per month per POS/cash loan | Monthly balance snapshots |
| `credit_card_balance` | one row per month per credit card | Monthly balance snapshots |
| `installments_payments` | one row per payment | Repayment history of earlier credits |

`application_test` has no labels. After feature engineering it is stored as `test_features.parquet` and is used **only** for drift monitoring (PSI), never for evaluation.

- **Target:** `TARGET = 1` means the client had payment difficulties (late payment of more than the stated number of days on at least one of the first instalments); `0` otherwise. The label is observed only for loans that were actually granted, so the data carries selection bias (rejected applicants are absent).
- **Class imbalance:** about 8% positives in `application_train` (per the Kaggle data description and common knowledge of this dataset; the exact rate is logged by `train.py` as the target rate and was not re-measured here). Metrics that are sensitive to imbalance (PR-AUC, Brier, business cost) are reported next to ROC-AUC.
- **Splits:** a holdout of 15% of `application_train` (`data.holdout_size`) is separated by `SK_ID_CURR` **before** any encoder, feature selector, HPO or threshold is fitted, with the seed from `params.yaml`. The ids are stored in `data/processed/holdout_ids.json` (DVC stage `split`). The remaining 85% is used for cross-validation, HPO and the final fit. The holdout is scored exactly once, in `src/models/train.py`.

## Personal and protected data

The dataset is pseudonymous (no names, addresses or contact details) but contains personal and sensitive attributes:

| Field | Why it matters | Handling here |
|---|---|---|
| `CODE_GENDER` | Protected attribute (sex) | Excluded from model features (`features.exclude`), kept in the holdout frame for the fairness audit |
| `DAYS_BIRTH` | Age; protected attribute | Remains a model feature (age is a legitimate risk factor, see the model card); audited by age bin |
| `NAME_FAMILY_STATUS` | Marital status; sensitive and correlated with sex and age | Not excluded and not audited separately in the current code; this is a known gap |

Other fields (income, occupation, region, housing) can act as proxies for protected attributes. Excluding `CODE_GENDER` does not remove such proxy effects, which is why disparate impact and equal-opportunity gaps are measured on the holdout rather than assumed.

## Known anomalies and how they are handled

- `DAYS_EMPLOYED = 365243` (about 1000 years) is a placeholder for "not employed / not applicable" (per the Kaggle data description and community analyses, mostly non-working clients such as pensioners; not verified here). It is replaced by NaN and an indicator `DAYS_EMPLOYED_ANOM` is added (`src/data/preprocessor.py`). The same sentinel is replaced by NaN in the `DAYS_*` columns of `previous_application`.
- `XNA` and `Unknown` category values (for example in `CODE_GENDER`, `ORGANIZATION_TYPE`) are turned into missing values only in the application tables (`ApplicationPreprocessor.fix_anomalies`); the side tables are not cleaned for them (bureau tables pass through unchanged). Missing categorical values are then label-encoded as their own class (`__MISSING__`), so `CODE_GENDER` appears in the fairness CSVs as integer codes (for example 0/1/2), not as F/M/XNA.
- `DAYS_BIRTH` is stored as negative days before the application; the preprocessor takes the absolute value.
- Missing values are widespread. The original pre-audit OOF-era analysis (README, Hypothesis 2) found a higher share of missing application fields among defaulters (median 39.7% vs 28.1%); this was not re-run after the audit changes.
- Groups smaller than `fairness.min_group_size` (30) in the holdout (for example the very few `XNA` values of `CODE_GENDER`, if any) are reported but excluded from the disparate-impact and equal-opportunity metrics.

## Intended use and limitations

- Intended for research and for demonstrating an MLOps workflow (training, tracking, promotion gate, drift and fairness guardrails). It is not a production credit decision system.
- The data describes one lender's population and a past period. Scores should not be assumed to transfer to other markets, products or time periods.
- Because only granted loans have labels, the models learn the default risk of previously accepted clients, not of the whole applicant population.

## Maintenance

- Raw data is fixed by Kaggle; reproducibility is provided by `uv.lock`, `dvc.yaml`/`dvc.lock` and the seeded holdout split.
- After downloading the data, run `uv run dvc add data/raw` and commit the resulting `.dvc` file; this has not been done yet because the data is not stored on every development machine.
