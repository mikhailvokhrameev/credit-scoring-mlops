# План исправлений по аудиту credit-scoring-mlops

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести репозиторий с MLOps L0 до L1–L2: честная оценка на holdout без утечек, конфиг вместо констант, данные и пайплайн под DVC, lock-файл uv, тесты и CI с гейтом по метрике, guardrail-метрики (калибровка, fairness, дрифт) и документация, которая не расходится с кодом.

**Architecture:** Holdout по `SK_ID_CURR` выделяется отдельной стадией до любого `fit` и сохраняется отдельными parquet-файлами, поэтому код обучения физически не может его увидеть. Все настраиваемые значения читаются из `params.yaml` через `src/config.load_params()`. Обучение становится функцией `run_training(...)`: её вызывает и CLI, и smoke-тест на синтетической фикстуре, которую гоняет CI.

**Tech Stack:** Python 3.10, uv, pandas, scikit-learn, LightGBM/XGBoost/CatBoost, Optuna, MLflow (Registry с aliases), SHAP, DVC, pytest, ruff, GitHub Actions.

**Spec:** `/Users/mikhailvokhrameev/.claude/plans/compressed-twirling-phoenix.md` (аудит; разделы 2 «Дефекты» F1–F13 и 3 «План исправлений» п. 1–21). При исполнении план копируется в `docs/superpowers/plans/2026-10-17-audit-fixes.md`.

## Global Constraints

- Ветка `audit-fixes` от `main`, в `main` через PR.
- **Коммиты без `Co-Authored-By: Claude` и без любой AI-атрибуции**, в том числе в описании PR. Автор `mikhaivokhrameev`. Сообщения короткие, в повелительном наклонении, как в истории репозитория: `Add holdout split before feature selection`.
- `requires-python = "==3.10.*"`. Окружение только uv (`uv sync`, `uv run …`), `uv.lock` коммитится, conda/pip не используются.
- Ни один `fit` (энкодеры, отбор признаков, HPO, early stopping, порог) не видит holdout. Holdout оценивается ровно один раз, в `run_training`.
- Сиды, пути, стоимости, `top_n`, число фолдов, имя эксперимента и tracking URI задаются только в `params.yaml`. Литерал `42` или `10.0` в `src/` — дефект.
- Тесты работают без данных Kaggle, только на синтетической фикстуре из `tests/conftest.py` (500 строк).
- Отсутствующий признак на инференсе → `ValueError` с именами колонок. Нули не подставляются, исключения не глотаются.
- Лекции MLOps 2, 4, 5 (LLM) вне скоупа.

## Review Focus

1. **Группа без положительных (или без отрицательных) примеров в fairness-разрезе** (например, бин «60+» на маленьком holdout): ожидается `NaN` в `auc`/`good_approval_rate` этой группы и предупреждение в логе, а не падение обучения. Тест: Task 8, `test_fairness_degenerate_group`.
2. **Обучение на артефактах старого пайплайна** (нет `holdout_features.parquet` или `holdout_ids.json` после `git pull`): ожидается `FileNotFoundError` с текстом «run `dvc repro features` first», а не обучение на всём train. Тест: Task 7, `test_run_training_requires_holdout`.
3. **Вход инференса с лишними колонками, другим порядком или «сырыми» именами с пробелами**: ожидается корректный скор, а при отсутствии колонки `ValueError` с её именем. Тест: Task 3, `test_align_extra_and_reordered_columns`.
4. **Все trials HPO упали** (OOM, ошибка GPU): ожидается `RuntimeError` с текстом первой исходной ошибки, а не непрозрачное `ValueError: No trials are completed yet` от Optuna. Тест: Task 5, `test_all_trials_failed_raises_with_cause`.
5. **PSI на признаке с NaN, константой или пустыми бинами**: ожидается конечное неотрицательное число. Тест: Task 9, `test_psi_nan_and_constant`.

---

### Task 1: Переход на uv и каркас тестов

**Files:**
- Create: `pyproject.toml`, `.python-version`, `uv.lock`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_env.py`
- Delete: `requirements.txt`

**Interfaces:**
- Produces: фикстура `synthetic_df(n: int = 500, seed: int = 0) -> pd.DataFrame` (обычная функция, плюс pytest-фикстура `synth` = `synthetic_df()`), колонки `SK_ID_CURR` (int, уникальный), `TARGET` (0/1, ~20% единиц), `EXT_SOURCE_1..3` (float, ~10% NaN в `EXT_SOURCE_1`), `DAYS_BIRTH` (int, положительные дни, возраст 20–69), `CODE_GENDER` (0/1), `AMT_CREDIT`, `NOISE_0..NOISE_9`. TARGET = Bernoulli(sigmoid(−2·mean(EXT_SOURCE) + …)), сигнал такой, что LogReg даёт AUC ≥ 0.75. Фикстура `synthetic_data_dir(tmp_path) -> Path`: пишет `processed/{train,holdout,test}_features.parquet`, `selected_features.json`, `holdout_ids.json` и `feature_importance.csv` в layout из Task 6, holdout 20%.

- [ ] **Step 1:** `git checkout -b audit-fixes`. В рабочем conda-окружении снять версии: `conda activate credit-scoring && pip freeze > /tmp/freeze.txt`. Если окружения нет, взять версии из последнего MLflow-рана (`mlflow.source.*` / `requirements.txt` артефакта модели).
- [ ] **Step 2:** grep по `src/` и `notebooks/` на `lime|dtreeviz|interpret|imblearn|structlog|plotly|ipywidgets`. Ненайденные пакеты не переносить.
- [ ] **Step 3:** Написать `pyproject.toml`: `[project] name="credit-scoring-mlops"`, `requires-python="==3.10.*"`, `dependencies` = pandas, pyarrow, scikit-learn, lightgbm, xgboost, catboost, optuna, optuna-dashboard, mlflow, dvc, shap, matplotlib, seaborn, scipy, pyyaml, kaggle, tabulate (нужен `to_markdown` в `compare_models.py`) с нижними границами `>=` из freeze. `[dependency-groups] dev = [pytest, ruff]`, `notebooks = [jupyterlab, ydata-profiling, nbconvert, quarto]`. `[tool.pytest.ini_options] testpaths=["tests"]`, `markers=["smoke"]`. `.python-version` = `3.10`.
- [ ] **Step 4:** `uv lock`, затем `uv sync --group dev`. Удалить `requirements.txt`.
- [ ] **Step 5:** Написать `tests/conftest.py` (интерфейс выше) и `tests/test_env.py`:

```python
def test_heavy_imports():
    import lightgbm, xgboost, catboost, mlflow, shap, optuna  # noqa: F401

def test_synthetic_df_shape(synth):
    assert len(synth) == 500 and synth["SK_ID_CURR"].is_unique
    assert 0.1 < synth["TARGET"].mean() < 0.3
```

- [ ] **Step 6:** `uv run pytest -q`. Ожидается: 2 passed. `uv lock --check`: без расхождений.
- [ ] **Step 7:** Коммит `Migrate environment from conda to uv` (pyproject, .python-version, uv.lock, tests/, удаление requirements.txt).

---

### Task 2: `params.yaml` и `src/config.py` (configuration debt, F9)

**Files:**
- Create: `params.yaml`, `src/__init__.py` (если нет), `src/config.py`, `tests/test_config.py`
- Modify: `src/features/pipeline.py:34,113,118,227`, `src/features/pipeline_base.py:13`, `src/data/preprocessor.py:55,94,116,183`, `src/evaluation/compare_models.py:7-9`, `src/evaluation/explainer.py` (сид сэмплирования)

**Interfaces:**
- Produces: `ROOT: Path` (= `Path(__file__).resolve().parents[1]`), `load_params(path: Path | None = None) -> dict` (по умолчанию `ROOT / "params.yaml"`; при отсутствии обязательной секции `KeyError` с её именем), `stable_hash(obj: Any) -> str` (sha256 от `json.dumps(obj, sort_keys=True, default=str)`, hex). Все модули ниже берут значения из `load_params()`.

`params.yaml` (значения фиксированы):

```yaml
seed: 42
data:
  holdout_size: 0.15
features:
  top_n: 300
  cardinality_threshold: 10
  exclude: [CODE_GENDER]        # см. Task 8 / model card
  protected: [CODE_GENDER, DAYS_BIRTH]
cv:
  n_splits: 5
hpo:
  n_splits: 3
  trials: 30
  timeout: 3600
train:
  early_stopping_rounds: 100
  early_stopping_fraction: 0.1
business:
  fn_cost: 10.0
  fp_cost: 1.0
mlflow:
  experiment: Home_Credit_Default_Risk
  tracking_uri: sqlite:///mlflow.db
monitoring:
  psi_alert: 0.25
  psi_top_k: 20
fairness:
  di_min: 0.8
gate:
  auc_epsilon: 0.005
  fixture_auc_min: 0.70
```

- [ ] **Step 1: Failing tests** в `tests/test_config.py`:

```python
def test_load_default_params():
    p = load_params()
    assert p["seed"] == 42 and p["features"]["top_n"] == 300

def test_override_file(tmp_path):
    src = yaml.safe_load((ROOT / "params.yaml").read_text())
    src["features"]["top_n"] = 200; src["business"]["fn_cost"] = 5.0
    f = tmp_path / "params.yaml"; f.write_text(yaml.safe_dump(src))
    p = load_params(f)
    assert p["features"]["top_n"] == 200 and p["business"]["fn_cost"] == 5.0

def test_missing_section_names_key(tmp_path):
    f = tmp_path / "p.yaml"; f.write_text("seed: 1\n")
    with pytest.raises(KeyError, match="features"):
        load_params(f)

def test_stable_hash_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

def test_no_hardcoded_seed_in_src():
    hits = [p for p in (ROOT / "src").rglob("*.py")
            if re.search(r"random_state\s*=\s*42|seed\s*=\s*42|fn_cost\s*=\s*10", p.read_text())]
    assert hits == []
```

- [ ] **Step 2:** `uv run pytest tests/test_config.py -v`. Ожидается FAIL (`ModuleNotFoundError: src.config`).
- [ ] **Step 3:** Реализовать `src/config.py`. Обязательные секции: `seed, data, features, cv, hpo, train, business, mlflow, monitoring, fairness, gate`. Заменить `ROOT = Path.cwd()` в обоих пайплайнах на `from src.config import ROOT`. Протянуть `cardinality_threshold` и `seed` параметрами в `reduce_cardinality`/`process_side_table`/`target_encode_cv` (вызывающий передаёт значения из конфига). В `compare_models.py` брать `experiment` и `tracking_uri` из конфига. В пайплайнах `mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])` перед `set_experiment`. Модели и `train.py` правятся в Tasks 3–7; `test_no_hardcoded_seed_in_src` до Task 7 помечен `@pytest.mark.xfail(strict=False, reason="models/train migrated in Tasks 3-7")`, в Task 7 маркер снимается.
- [ ] **Step 4:** `uv run pytest tests/test_config.py -v`. Ожидается: 4 passed, 1 xfailed.
- [ ] **Step 5:** Коммит `Move tunable constants to params.yaml`.

---

### Task 3: Обёртки моделей: общая схема, падение на пропусках, иммутабельные params (F6, F7)

**Files:**
- Modify: `src/models/base.py`, `src/models/lgbm_model.py`, `src/models/xgb_model.py`, `src/models/catboost_model.py`, `src/models/logreg_model.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces в `BaseModel`:
  - `_sanitize_columns(X) -> pd.DataFrame` и `_align_features(X) -> pd.DataFrame` переезжают в базовый класс, копии в четырёх наследниках удаляются. `_align_features` бросает `ValueError(f"Missing features at inference: {sorted(missing)}")` и возвращает `X[self.features_]`, лишние колонки отбрасываются.
  - `transform(X)` = sanitize + align (в базе, не `NotImplementedError`).
  - Атрибут класса `n_estimators_key: Optional[str]`: `"n_estimators"` у LGBM/XGB, `"iterations"` у CatBoost, `None` у LogReg.
  - После `fit` атрибут `best_iteration_: Optional[int]` (из `best_iteration_` LGBM, `best_iteration` XGB, `get_best_iteration()` CatBoost; `None`, если eval_set не передан, и у LogReg).
  - Сигнатура `fit(X, y, eval_set=None, early_stopping_rounds: Optional[int] = None)`. Ранняя остановка включается только при наличии и `eval_set`, и `early_stopping_rounds`; `cross_validate` передаёт значение из `params["train"]["early_stopping_rounds"]`. В params модели этот ключ не кладётся.
- `fit` ни в одном наследнике не меняет `self.params`: работает с `params = dict(self.params)`. `mlflow.xgboost.autolog()` и `mlflow.sklearn.autolog()` из `fit` удаляются.

- [ ] **Step 1: Failing tests**, параметризованные по `[LogRegModel, LGBMModel, XGBModel, CatBoostModel]` с минимальными params (`n_estimators`/`iterations` = 50, сид из `load_params()`):

```python
def test_missing_feature_raises(cls, synth):
    X, y = features(synth), synth["TARGET"]
    m = cls(small_params(cls)).fit(X, y)
    with pytest.raises(ValueError, match="EXT_SOURCE_2"):
        m.predict_proba(X.drop(columns=["EXT_SOURCE_2"]))

def test_align_extra_and_reordered_columns(cls, synth):
    X, y = features(synth), synth["TARGET"]
    m = cls(small_params(cls)).fit(X, y)
    shuffled = X[X.columns[::-1]].assign(EXTRA_COL=1.0)
    np.testing.assert_allclose(m.predict_proba(shuffled), m.predict_proba(X))

def test_fit_does_not_mutate_params(cls, synth):
    p = small_params(cls) | {"device": "cpu"}
    before = copy.deepcopy(p)
    m = cls(p); m.fit(features(synth), synth["TARGET"])
    assert m.params == before

def test_best_iteration_set_with_eval_set(cls, synth):
    ...  # для cls.n_estimators_key is not None: best_iteration_ is int и 0 < it <= 50; иначе None
```

`features(df)` в conftest = все колонки, кроме `SK_ID_CURR`, `TARGET` и `CODE_GENDER`.

- [ ] **Step 2:** `uv run pytest tests/test_models.py -v`. Ожидается FAIL: XGB мутирует params, предсказание без колонки не бросает исключение.
- [ ] **Step 3:** Реализовать интерфейсы выше. В XGB: `params.setdefault("scale_pos_weight", neg/pos)`, `device` → `"cuda"`/`"cpu"` и `tree_method="hist"` в локальной копии.
- [ ] **Step 4:** `uv run pytest tests/test_models.py -v`. Ожидается: все passed.
- [ ] **Step 5:** Коммит `Fail on missing inference features and stop mutating model params`.

---

### Task 4: CV без оптимистичного early stopping (F3, F10)

**Files:**
- Modify: `src/models/base.py` (`cross_validate`, `optimize_threshold`)
- Test: `tests/test_cv.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `best_iteration_`, `n_estimators_key` из Task 3.
- Produces:
  - `@dataclass CVResult: oof: np.ndarray; models: list[BaseModel]; fold_auc: list[float]; best_iterations: list[int]` и свойство `mean_auc -> float`.
  - `BaseModel.cross_validate(X, y, n_splits: int, seed: int, early_stopping_rounds: int, es_fraction: float) -> CVResult`. В каждом фолде train-часть делится `train_test_split(stratify=y_tr, test_size=es_fraction, random_state=seed)` на `fit`/`es`. `eval_set=(X_es, y_es)` и OOF считается на нетронутом `X_val`. Для `n_estimators_key is None` внутреннее разбиение не делается. Метрики фолдов логируются в MLflow, только если есть активный run.
  - `BaseModel.refit_params(cv: CVResult) -> dict`: копия `self.params`, в которой `n_estimators_key` = `int(np.median(cv.best_iterations)) + 1`. Если ключа нет, params возвращаются без изменений.
  - `optimize_threshold(y_true, y_proba, fn_cost: float, fp_cost: float) -> dict` (без дефолтов стоимостей). F1-порог: `f1_scores[:-1]` перед `argmax`, ветка `else 0.5` удаляется.
  - `evaluate_threshold(y_true, y_proba, threshold: float, fn_cost: float, fp_cost: float) -> dict` с ключами `business_loss, fn, fp, approval_rate`, где approval = `y_proba < threshold`.

- [ ] **Step 1: Failing tests**:

```python
def test_cv_reproducible(synth):              # LGBM, seed из конфига
    a = LGBMModel(p).cross_validate(X, y, 3, 42, 20, 0.1)
    b = LGBMModel(p).cross_validate(X, y, 3, 42, 20, 0.1)
    np.testing.assert_array_equal(a.oof, b.oof)

def test_cv_val_fold_not_used_for_early_stopping(synth, monkeypatch):
    # обёртка над LGBMModel.fit запоминает индексы eval_set; ни один из них не входит в val_idx своего фолда
    ...

def test_refit_params_uses_median_best_iteration():
    cv = CVResult(oof=np.zeros(3), models=[], fold_auc=[.7]*3, best_iterations=[10, 30, 20])
    assert LGBMModel({"n_estimators": 2000}).refit_params(cv)["n_estimators"] == 21
    assert LogRegModel({"C": 1}).refit_params(cv) == {"C": 1}
```

`tests/test_metrics.py`:

```python
def test_ks_perfect_separation():
    assert compute_ks_statistic(np.array([0,0,1,1]), np.array([.1,.2,.8,.9])) == 1.0

def test_gini_from_auc():
    m = compute_all_metrics(np.array([0,1,0,1]), np.array([.1,.9,.2,.8]))
    assert m["roc_auc"] == 1.0 and m["gini"] == 1.0

def test_f1_threshold_is_real_threshold():
    y = np.array([0,0,1,1]); p = np.array([.1,.4,.35,.8])
    t = LogRegModel().optimize_threshold(y, p, 10.0, 1.0)["best_f1_threshold"]
    assert t in p

def test_evaluate_threshold_counts():
    r = evaluate_threshold(np.array([1,0,1,0]), np.array([.9,.6,.2,.1]), .5, 10.0, 1.0)
    assert (r["fn"], r["fp"], r["business_loss"], r["approval_rate"]) == (1, 1, 11.0, .5)
```

- [ ] **Step 2:** `uv run pytest tests/test_cv.py tests/test_metrics.py -v`. Ожидается FAIL (`CVResult` не определён).
- [ ] **Step 3:** Реализовать. `evaluate_threshold` положить в `src/evaluation/metrics.py`, рядом с `compute_all_metrics`.
- [ ] **Step 4:** Те же тесты. Ожидается: passed.
- [ ] **Step 5:** Коммит `Hold out early-stopping data inside CV folds`.

---

### Task 5: Optuna: изоляция study и честные ошибки (F2, F5)

**Files:**
- Modify: `src/models/tuner.py`
- Test: `tests/test_tuner.py`

**Interfaces:**
- Consumes: `CVResult.mean_auc` (Task 4), `stable_hash` (Task 2).
- Produces:
  - `data_fingerprint(X: pd.DataFrame, y: pd.Series) -> str`: sha256 от `pd.util.hash_pandas_object(X, index=False).values` + `list(X.columns)` + хэша `y`.
  - `make_study_name(model_name: str, X, y, params: dict) -> str` = `f"{model_name}_{data_fingerprint(X, y)[:8]}_{stable_hash({k: params[k] for k in ('seed','features','hpo','train')})[:8]}"`.
  - `OptunaHPOTuner(model_class, storage: str, study_name: str, seed: int, base_params: dict)`: без pruner (HyperbandPruner удалён), `TPESampler(seed=seed)`. `load_if_exists=True` остаётся: имя уже уникально для данных и конфига, а возобновление прерванного HPO полезно.
  - `optimize(X, y, n_trials, timeout, cv_kwargs: dict) -> dict` возвращает `best_params | base_params`. Objective возвращает `model.cross_validate(...).mean_auc` напрямую, без чтения MLflow. Исключение: `logger.exception`, тег `status=failed` на nested run, затем re-raise. `study.optimize(..., catch=(Exception,))`. После цикла, если нет ни одного `TrialState.COMPLETE`, бросается `RuntimeError(f"All {n} HPO trials failed; first error: {msg}")`, где `msg` хранится в `trial.user_attrs["error"]`.

- [ ] **Step 1: Failing tests**:

```python
def test_study_name_changes_with_data_and_config(synth):
    X, y = features(synth), synth["TARGET"]; p = load_params()
    n1 = make_study_name("lgbm", X, y, p)
    assert make_study_name("lgbm", X.iloc[:-1], y.iloc[:-1], p) != n1
    p2 = copy.deepcopy(p); p2["features"]["top_n"] = 200
    assert make_study_name("lgbm", X, y, p2) != n1
    assert make_study_name("lgbm", X, y, p) == n1

def test_all_trials_failed_raises_with_cause(tmp_path, synth, monkeypatch):
    monkeypatch.setattr(LogRegModel, "cross_validate", lambda *a, **k: (_ for _ in ()).throw(MemoryError("boom-oom")))
    t = OptunaHPOTuner(LogRegModel, f"sqlite:///{tmp_path}/o.db", "s", 42, {})
    with pytest.raises(RuntimeError, match="boom-oom"):
        t.optimize(X, y, n_trials=2, timeout=60, cv_kwargs=...)

def test_optimize_returns_best_with_base_params(tmp_path, synth):
    ...  # LogReg, 2 trials: "C" в результате и base_params["device"] == "cpu"
```

- [ ] **Step 2:** `uv run pytest tests/test_tuner.py -v`. Ожидается FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Тесты зелёные.
- [ ] **Step 5:** Коммит `Scope Optuna studies to data and config, surface trial failures`.

---

### Task 6: Holdout до отбора признаков и энкодеров (F1)

**Files:**
- Create: `src/data/split.py`, `tests/test_no_leakage.py`
- Modify: `src/features/pipeline.py` (`run`, `_select_features_ridge`, `_save_results`), `src/features/pipeline_base.py`, `src/data/loader.py` (`__main__` для DVC-стадии `prepare`)

**Interfaces:**
- Produces:
  - `make_holdout_ids(ids: pd.Series, target: pd.Series, size: float, seed: int) -> np.ndarray`: стратифицированный `train_test_split`, отсортированный массив `SK_ID_CURR`.
  - `save_ids(ids, path: Path) -> None` и `load_ids(path: Path) -> np.ndarray` (JSON-список int).
  - CLI `python -m src.data.split`: читает `data/cache/application_train.parquet` (через `DataLoader`), пишет `data/processed/holdout_ids.json`.
  - `select_features_ridge(X: pd.DataFrame, y: pd.Series, top_n: int) -> tuple[list[str], pd.DataFrame]`: чистая функция модуля `pipeline.py`, возвращает (выбранные колонки, таблицу важностей по убыванию). Метод пайплайна вызывает её только на строках `fit` (train без holdout).
  - Layout выхода `data/processed/` (на нём держатся Task 1 fixture и Task 7): `train_features.parquet` (train без holdout), `holdout_features.parquet`, `test_features.parquet` (Kaggle test без TARGET), `selected_features.json` (без колонок из `features.exclude`), `feature_importance.csv`, `holdout_ids.json`. Колонки из `features.protected` сохраняются во всех трёх parquet, даже если не отобраны в признаки. Базовый пайплайн пишет аналогичную тройку в `data/processed_base/`.
- Правило энкодеров: `fit_ids = setdiff1d(train_ids, holdout_ids)`. В `process_side_table` передаётся `fit_ids` вместо `train_ids`, так что строки holdout уходят в «transform-часть». `fit_transform_categorical(fit_part, rest)`, где `rest = concat(train_raw[holdout], test_raw)`. Тот же приём в `pipeline_base.py`.

- [ ] **Step 1: Failing tests**:

```python
def test_holdout_stratified_and_deterministic(synth):
    a = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    assert len(a) == 100 and np.array_equal(a, make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42))
    assert abs(synth.set_index("SK_ID_CURR").loc[a, "TARGET"].mean() - synth.TARGET.mean()) < 0.03

def test_process_side_table_encoders_ignore_holdout():
    side = pd.DataFrame({"SK_ID_CURR": [1,1,2,3], "CAT": ["a","a","b","ONLY_IN_HOLDOUT"]})
    out, cat_cols = process_side_table(side, fit_ids=np.array([1,2]), preprocessor=PassThroughPreprocessor(),
                                       table_name="t", cardinality_threshold=10)
    assert not any("ONLY_IN_HOLDOUT" in c for c in cat_cols)

def test_label_encoder_fit_without_holdout():
    fit = pd.DataFrame({"C": ["x","y"]}); rest = pd.DataFrame({"C": ["z"]})
    _, r = ApplicationPreprocessor().fit_transform_categorical(fit, rest)
    # "z" не встречался в fit, поэтому кодируется как UNKNOWN (последний класс)
    assert r["C"].iloc[0] == 2

def test_ridge_selection_independent_of_holdout_rows(synth):
    hold = make_holdout_ids(synth.SK_ID_CURR, synth.TARGET, 0.2, 42)
    fit = synth[~synth.SK_ID_CURR.isin(hold)]
    cols1, _ = select_features_ridge(features(fit), fit.TARGET, 5)
    poisoned = synth.copy(); poisoned.loc[poisoned.SK_ID_CURR.isin(hold), "NOISE_0"] = poisoned.TARGET * 100
    fit2 = poisoned[~poisoned.SK_ID_CURR.isin(hold)]
    assert select_features_ridge(features(fit2), fit2.TARGET, 5)[0] == cols1

def test_excluded_features_not_selected(...):  # CODE_GENDER нет в selected_features.json, но есть в holdout parquet
```

- [ ] **Step 2:** `uv run pytest tests/test_no_leakage.py -v`. Ожидается FAIL (`src.data.split` нет; `process_side_table` не принимает `fit_ids`).
- [ ] **Step 3:** Реализовать. Сигнатура: `process_side_table(df, fit_ids, preprocessor, table_name, cardinality_threshold: int)`. `FeatureEngineeringPipeline.run()` читает `holdout_ids.json`; при его отсутствии `FileNotFoundError("run `python -m src.data.split` first")`.
- [ ] **Step 4:** Тесты зелёные. Ручная проверка на реальных данных (нужен Kaggle): `uv run python -m src.data.split && uv run python -m src.features.pipeline`. В `data/processed/` должны появиться три parquet, а `len(holdout)/(len(train)+len(holdout))` ≈ 0.15.
- [ ] **Step 5:** Коммит `Add holdout split before feature selection`.

---

### Task 7: `train.py`: holdout один раз, refit, конфиг, Registry challenger, `--base` (F4, F11, F12)

**Files:**
- Modify: `src/models/train.py` (переписать)
- Delete: `src/models/train_base.py`
- Test: `tests/test_train_smoke.py`
- Modify: `tests/test_config.py` (снять `xfail`)

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces:
  - `run_training(model_name: str, params: dict, *, data_dir: Path, artifact_dir: Path, trials: int, device: str = "cpu", base: bool = False, register: bool = True) -> dict[str, float]`. Возвращает метрики с префиксами `final_oof_` и `holdout_`.
  - Порядок шагов: load (`base` → `data/processed_base/`, иначе `data/processed/`; без holdout-файла `FileNotFoundError("... run `dvc repro features` first")`) → HPO (`make_study_name`, storage `sqlite:///{ROOT}/optuna.db`; при `trials == 0` пропускается) → `cross_validate` на train → `optimize_threshold` на OOF → `refit_params` → final `fit` на всём train → **единственный** `predict_proba(holdout)` → `compute_all_metrics(prefix="holdout_")` + `evaluate_threshold(best_business_threshold)` с префиксом `holdout_` → SHAP (Task 10) → `log_model(registered_model_name=f"credit_scoring_{model_name}{'_base' if base else ''}")` + alias `challenger` (`MlflowClient().set_registered_model_alias`) → `artifact_dir/f"{model_name}_model"/"metrics.json"` (плоский JSON метрик для DVC).
  - Имена MLflow-ранов сохраняются (`{model}` / `{model}_base`, nested `HPO`, `Final_Production`), чтобы `compare_models.py` продолжал работать.
  - `main()`: argparse `--model`, `--trials` (default из `hpo.trials`), `--folds` (default из `cv.n_splits`), `--device`, `--base`, `--params PATH`. CLI-флаги переопределяют конфиг.

- [ ] **Step 1: Failing test** (`@pytest.mark.smoke`):

```python
def test_run_training_logreg_smoke(synthetic_data_dir, tmp_path, monkeypatch):
    p = load_params(); p["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"; p["cv"]["n_splits"] = 3
    m = run_training("logreg", p, data_dir=synthetic_data_dir, artifact_dir=tmp_path/"art", trials=0)
    assert m["holdout_roc_auc"] >= p["gate"]["fixture_auc_min"]          # гейт по метрике на фикстуре
    assert {"final_oof_roc_auc", "holdout_business_loss", "holdout_brier"} <= m.keys()  # brier появится в Task 8
    assert json.loads((tmp_path/"art/logreg_model/metrics.json").read_text())["holdout_roc_auc"] == m["holdout_roc_auc"]
    v = MlflowClient(p["mlflow"]["tracking_uri"]).get_model_version_by_alias("credit_scoring_logreg", "challenger")
    assert v is not None

def test_run_training_requires_holdout(synthetic_data_dir, tmp_path):
    (synthetic_data_dir/"processed/holdout_features.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="dvc repro"):
        run_training("logreg", load_params(), data_dir=synthetic_data_dir, artifact_dir=tmp_path, trials=0)

def test_holdout_predicted_once(...):
    # monkeypatch LogRegModel.predict_proba считает вызовы на кадрах с SK_ID из holdout; == 1
```

Пока не сделан Task 8, проверку `holdout_brier` в первом тесте держать закомментированной; Task 8 её раскомментирует.

- [ ] **Step 2:** `uv run pytest tests/test_train_smoke.py -v`. Ожидается FAIL (`run_training` нет).
- [ ] **Step 3:** Реализовать, удалить `train_base.py`, снять `xfail` в `test_no_hardcoded_seed_in_src`.
- [ ] **Step 4:** `uv run pytest -q`. Ожидается: всё passed, включая `test_no_hardcoded_seed_in_src`.
- [ ] **Step 5:** Коммит `Evaluate on holdout once and register challenger model`.

---

### Task 8: Guardrail-метрики: калибровка и fairness на holdout

**Files:**
- Create: `src/evaluation/fairness.py`, `tests/test_fairness.py`
- Modify: `src/evaluation/metrics.py`, `src/models/train.py` (блок holdout-оценки), `tests/test_metrics.py`, `tests/test_train_smoke.py`

**Interfaces:**
- Produces:
  - В `compute_all_metrics` добавляется ключ `brier` (`brier_score_loss`).
  - `reliability_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame` с колонками `bin, mean_pred, frac_pos, count` (квантильные бины) и `plot_reliability(table, path: Path) -> Path`.
  - `age_groups(days_birth: pd.Series) -> pd.Series`: годы = `days_birth / 365` (ровно 365, так же, как в тесте), бины `pd.cut(..., [0,30,40,50,60,200], right=False)` с метками `"<30","30-39","40-49","50-59","60+"`.
  - `fairness_report(y_true, y_proba, groups: pd.Series, threshold: float) -> pd.DataFrame`: индекс — группа, колонки `n, approval_rate, good_approval_rate, auc`. Approval = `proba < threshold`; `good_approval_rate` = P(approved | y=0) (equal opportunity для хороших заёмщиков). Для вырожденной группы `auc = NaN` плюс warning.
  - `fairness_summary(report) -> dict`: `{"di_ratio": min/max approval_rate, "eo_gap": max-min good_approval_rate}`, NaN игнорируются.
  - В `run_training` метрики логируются как `holdout_fairness_{attr}_di_ratio` / `_eo_gap` для `attr ∈ {gender, age}`, плюс `holdout_fairness_min_di` (минимум по атрибутам, его читает Task 11). Таблицы и reliability-png идут в артефакты `guardrails/`. Тег `guardrails="brier,fairness"` помечает метрики как мониторящиеся, но не оптимизируемые.

- [ ] **Step 1: Failing tests**:

```python
def test_brier_known():
    assert compute_all_metrics(np.array([0,1]), np.array([0.,1.]))["brier"] == 0.0

def test_disparate_impact_four_fifths():
    y = np.zeros(8); p = np.array([.1,.1,.1,.9, .1,.9,.9,.9]); g = pd.Series(list("aaaabbbb"))
    s = fairness_summary(fairness_report(y, p, g, .5))
    assert s["di_ratio"] == pytest.approx(0.25 / 0.75)

def test_fairness_degenerate_group(caplog):
    y = np.array([0,1,0,0]); p = np.array([.1,.8,.2,.3]); g = pd.Series(["a","a","b","b"])
    r = fairness_report(y, p, g, .5)
    assert np.isnan(r.loc["b", "auc"]) and "b" in caplog.text

def test_age_groups_edges():
    assert list(age_groups(pd.Series([29*365, 30*365, 61*365]))) == ["<30", "30-39", "60+"]
```

В smoke-тест Task 7 добавить (раскомментировать) `holdout_brier` и проверку наличия `holdout_fairness_min_di`.
- [ ] **Step 2:** Тесты падают (`ModuleNotFoundError: src.evaluation.fairness`).
- [ ] **Step 3:** Реализовать и встроить в `run_training`.
- [ ] **Step 4:** `uv run pytest -q`. Всё зелёное.
- [ ] **Step 5:** Коммит `Add calibration and fairness guardrail metrics`.

---

### Task 9: Мониторинг дрейфа (PSI)

**Files:**
- Create: `src/monitoring/__init__.py`, `src/monitoring/drift.py`, `tests/test_drift.py`
- Modify: `src/models/train.py`

**Interfaces:**
- Produces:
  - `psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float`. Алгоритм ниже не следует однозначно из сигнатуры, поэтому фиксируется: края бинов — уникальные квантили `expected` (без NaN), крайние бины открыты (`-inf`, `+inf`), NaN — отдельный бин; доли клипаются снизу на `eps`; `sum((a - e) * ln(a / e))`.
  - `drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame` с колонками `feature, psi`, сортировка по убыванию.
  - В `run_training`: `psi_score` = PSI скора модели на train против `test_features.parquet`; `psi_max_feature` = максимум по топ-`monitoring.psi_top_k` признакам из `feature_importance.csv`. При `psi_score > monitoring.psi_alert`: тег `drift_alert=true` и `logger.warning`. Обучение не падает: Kaggle test здесь суррогат прода и служит только сигналом. `test_features.parquet` используется только для PSI и нигде не оценивается.

- [ ] **Step 1: Failing tests**:

```python
def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)

def test_psi_shift_detected():
    r = np.random.default_rng(0)
    assert psi(r.normal(size=5000), r.normal(1.0, size=5000)) > 0.25

def test_psi_nan_and_constant():
    e = np.array([1.0]*100 + [np.nan]*10); a = np.array([1.0]*50 + [np.nan]*60)
    v = psi(e, a); assert np.isfinite(v) and v >= 0
```

- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Реализовать и встроить. Фикстура Task 1 уже пишет `test_features.parquet`; в smoke-тест добавить `assert "psi_score" in m`.
- [ ] **Step 4:** `uv run pytest -q`. Зелёное.
- [ ] **Step 5:** Коммит `Add PSI drift monitoring against Kaggle test`.

---

### Task 10: SHAP для XGBoost (F8)

**Files:**
- Modify: `src/evaluation/explainer.py` (`_create_explainer`), `src/models/train.py` (удалить ветку `if model_name == "xgb"`; SHAP-блок вынести в функцию `log_explanations(model, X, xai_dir, seed) -> None`)
- Test: `tests/test_explainer.py`

**Interfaces:**
- Для `XGBClassifier`: `shap.TreeExplainer(model.get_booster())`. При исключении `logger.warning` и fallback на `shap.Explainer(lambda d: model.predict_proba(d)[:, 1], shap.maskers.Independent(self.X_train.sample(min(200, len(self.X_train)), random_state=seed)))`. Сид передаётся в `ModelExplainer.__init__(..., seed: int)`.

- [ ] **Step 1: Failing test**, параметризованный по всем 4 моделям: fit на фикстуре (50 деревьев), `ModelExplainer(m, X, seed=42, output_dir=tmp_path)`, `compute_global_shap(X.head(50))` → `shap_values` имеет форму `(50, n_features)` (для двухклассового вывода берётся класс 1), без исключений.
- [ ] **Step 2:** `uv run pytest tests/test_explainer.py -v`. Ожидается FAIL только для xgb (или PASS, если зафиксированная версия shap уже совместима: тогда остаётся только удалить skip в `train.py`).
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Зелёное.
- [ ] **Step 5:** Коммит `Enable SHAP explanations for XGBoost`.

---

### Task 11: Гейт продвижения challenger → champion

**Files:**
- Create: `src/models/promote.py`, `tests/test_promote.py`

**Interfaces:**
- Consumes: alias `challenger` и метрики `holdout_roc_auc`, `holdout_fairness_min_di` из рана версии модели (Tasks 7, 8).
- Produces: `promote(registered_name: str, params: dict, client: MlflowClient | None = None) -> bool`. Правило: у challenger есть `holdout_roc_auc` и `holdout_fairness_min_di ≥ fairness.di_min`, и (champion нет **или** `challenger_auc ≥ champion_auc − gate.auc_epsilon`). Если правило выполнено, alias `champion` ставится на версию challenger и возвращается `True`; иначе `logger.warning` с причиной и `False`. CLI `python -m src.models.promote --model xgb` завершается с кодом 1 при `False`.

- [ ] **Step 1: Failing tests** на временном sqlite-трекинге: регистрируются две версии с заданными метриками через `mlflow.log_metrics` и `register_model`:
  - `test_first_challenger_becomes_champion` (DI 0.9).
  - `test_worse_challenger_rejected` (champion 0.78, challenger 0.77, eps 0.005) → `False`, champion не изменился.
  - `test_unfair_challenger_rejected` (AUC выше, DI 0.6) → `False`.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** Зелёное.
- [ ] **Step 5:** Коммит `Add champion promotion gate`.

---

### Task 12: DVC: данные и пайплайн

**Files:**
- Create: `.dvc/` (через `dvc init`), `dvc.yaml`, `data/raw.dvc`, `dvc.lock` (после первого `repro`)
- Modify: `.gitignore` (проверить, что `data/` игнорируется, а `!data/**/*.dvc` и `dvc.lock` нет)

**Interfaces:**
- Стадии `dvc.yaml`. Код стадии обязательно указывается в `deps` (ловушка из MLOps Л3).
  - `prepare`: `python -m src.data.loader`; deps `data/raw`, `src/data/loader.py`; outs `data/cache`.
  - `split`: `python -m src.data.split`; deps `data/cache/application_train.parquet`, `src/data/split.py`; params `seed`, `data.holdout_size`; outs `data/processed/holdout_ids.json`. Эта стадия пишет файл внутри `data/processed`, поэтому стадия `features` объявляет outs поштучно, а не весь каталог.
  - `features`: `python -m src.features.pipeline`; deps `data/cache`, `data/processed/holdout_ids.json`, `src/features`, `src/data`; params `seed`, `features`; outs — пять файлов из layout Task 6, кроме `holdout_ids.json`.
  - `features_base`: `python -m src.features.pipeline_base`; аналогично, outs `data/processed_base`.
  - `train` (`foreach: [logreg, lgbm, xgb, catboost]`): `python -m src.models.train --model ${item}`; deps выходы `features`, `src/models`, `src/evaluation`, `src/monitoring`, `src/config.py`; params `seed, cv, hpo, train, business, mlflow, monitoring, fairness`; outs `artifacts/${item}_model/model.joblib`; metrics `artifacts/${item}_model/metrics.json` (`cache: false`).
- `dvc add data/raw` создаёт `data/raw.dvc`. Remote не настраивается (решение пользователя, см. Caveats в CLAUDE.md).

- [ ] **Step 1:** `uv run dvc init && uv run dvc add data/raw`, написать `dvc.yaml`.
- [ ] **Step 2:** `uv run dvc dag`. Ожидается DAG `prepare → split → features/features_base → train@*` без ошибок.
- [ ] **Step 3:** `uv run dvc repro --dry`: перечислены все стадии. Полный `uv run dvc repro` идёт на реальных данных (часы на GPU), его запускает пользователь; затем повторный `dvc repro` должен ответить «up to date», а `touch src/features/bureau.py && dvc status` — показать `features` и `train@*` как changed.
- [ ] **Step 4:** Коммит `Track data and pipeline with DVC` (`.dvc/`, `dvc.yaml`, `data/raw.dvc`, `.gitignore`; `dvc.lock` отдельным коммитом после полного repro).

---

### Task 13: Харнес: ruff, `make check`, CI с гейтом

**Files:**
- Create: `Makefile`, `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (`[tool.ruff]`: `line-length = 120`, `select = ["E","F","I"]`, exclude `notebooks`)

**Interfaces:**
- `make check` = `uv run ruff check src tests && uv run pytest -q`. Smoke-train и гейт по AUC на фикстуре уже входят в pytest (Task 7). `make lint`, `make test`, `make smoke` (`pytest -m smoke`).
- CI: `on: [push, pull_request]`, `ubuntu-latest`, `actions/checkout@v4`, `astral-sh/setup-uv@v5` с `enable-cache: true`, `uv sync --locked --group dev` (падает при устаревшем lock), `make check`. Гейт «holdout-AUC ≥ champion − ε» для реальной модели выполняет `promote.py` локально после `dvc repro`, а не CI: в CI нет данных Kaggle.

- [ ] **Step 1:** `uv run ruff check src tests --fix`, оставшиеся замечания исправить вручную.
- [ ] **Step 2:** `make check`. Ожидается: ruff без ошибок, pytest всё passed.
- [ ] **Step 3:** Push ветки, открыть draft PR, убедиться, что workflow зелёный (`gh pr checks`).
- [ ] **Step 4:** Коммит `Add make check and CI workflow`.

---

### Task 14: Документация и слой памяти (F13)

**Files:**
- Create: `docs/datasheet.md`, `docs/model_card.md`, `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- `docs/datasheet.md`: происхождение (Kaggle Home Credit Default Risk), лицензия Kaggle competition rules, 7 таблиц, PII и защищённые поля (`CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`), аномалии (`DAYS_EMPLOYED=365243`, `XNA`), дисбаланс ~8%, holdout 15% по `SK_ID_CURR`.
- `docs/model_card.md`: назначение, метрики на holdout (из MLflow после `dvc repro`), fairness по полу и возрасту, решение об исключении `CODE_GENDER` (`features.exclude`), обоснование порога через `fn_cost/fp_cost = 10/1`, guardrail-метрики, error budget: «если ROC-AUC на свежей размеченной когорте < champion − 0.02 или PSI скора > 0.25 два периода подряд → откат на предыдущий champion» (урок Zillow), ограничения.
- README: блок установки → `uv sync` (+ `--group notebooks`), все команды через `uv run`, `train_base` → `--base`, uv в Tech Stack, URL клонирования → `credit-scoring-mlops.git`, «Model Registry» привязать к `promote.py`, «bank-grade reproducibility» → «reproducible (uv.lock + DVC + holdout)», «All 5 hypotheses were confirmed» → «4 confirmed, 1 partially», H1 переформулировать (монотонность по децилям, точное перестановочное p = 1/10! ≈ 2.8e-7, вывод про агрегаты, а не про отдельного заёмщика), таблицы результатов пометить «OOF, до исправления утечки (F1)» до пересчёта на holdout.
- `CLAUDE.md` — черновик из аудита (п. 18) **с одной правкой**: каждую команду перед коммитом прогнать вручную и оставить только работающие.

- [ ] **Step 1:** Написать datasheet и model card. Цифры holdout оставить плейсхолдерами `<после dvc repro>`, только если полный прогон ещё не сделан, и явно сказать об этом пользователю.
- [ ] **Step 2:** Обновить README по списку выше.
- [ ] **Step 3:** Прогнать каждую команду из черновика `CLAUDE.md` (`uv sync`, `uv run pytest -q`, `make check`, `uv run dvc dag`, `uv run python -m src.models.train --help`, `uv run mlflow ui --help`), записать `CLAUDE.md`.
- [ ] **Step 4:** Сохранить правило «коммиты без AI-атрибуции» в память Claude (feedback-память + строка в `MEMORY.md`).
- [ ] **Step 5:** `make check` зелёный. Коммит `Add datasheet, model card and CLAUDE.md; fix README claims`.
- [ ] **Step 6 (пользователь, на реальных данных):** `uv run dvc repro`, затем `uv run python -m src.models.promote --model <best>`. Перенести holdout-цифры в README и model card, закоммитить `dvc.lock` и документы (`Report holdout metrics after leakage fix`). Ожидается holdout-AUC на 0.002–0.01 ниже прежних OOF-цифр.

---

## Проверка всего результата

1. `rm -rf .venv && uv sync --locked --group dev && make check`: зелёное; `uv lock --check` чистый.
2. CI на PR зелёный.
3. `uv run dvc repro` с нуля воспроизводит `data/processed/*` и `artifacts/*/metrics.json`. Повторный запуск: «up to date».
4. `features.top_n: 200` в `params.yaml` → `dvc repro` без правок кода; в MLflow новый `top_n` и новое имя Optuna-study.
5. В MLflow у `Final_Production` есть `holdout_roc_auc`, `holdout_brier`, `holdout_fairness_min_di`, `psi_score`. Модель в Registry с alias `challenger`, после гейта — `champion`.
6. `git log main..audit-fixes --format=%B | grep -ci "co-authored-by\|claude"` → `0`.

## Исполнение

Рекомендуется **subagent-driven** (superpowers:subagent-driven-development): 14 задач с цепочкой интерфейсов (CVResult → tuner → run_training → fairness → promote), а ошибка в любой из них снова незаметно вносит утечку в метрики. Поэтому нужен независимый ревьюер на каждую задачу. Альтернатива — native (superpowers:executing-plans) с одним финальным ревью: так быстрее и дешевле.

Решения, принятые по умолчанию (поменять при ревью плана):
- `data.holdout_size = 0.15` (в аудите было 15–20%);
- `features.exclude = [CODE_GENDER]`: пол исключается из признаков, но остаётся для fairness-аудита, а возраст остаётся признаком;
- DVC remote не настраивается;
- CalibratedClassifierCV не добавляется (YAGNI): Brier и reliability только логируются как guardrail.
