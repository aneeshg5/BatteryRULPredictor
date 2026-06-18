## Phase 0 — Bootstrap — COMPLETE

**Date:** 2026-06-17

### What Was Built
- Initialized git repo, set default branch to `main`
- Moved paper transcription to `docs/paper_transcription.md`
- Added `.gitignore`, placeholder `README.md`

### Issues Encountered
- None

### Next Phase
Phase 1 — Project Scaffold & Environment

## Phase 1 — Project Scaffold & Environment — COMPLETE

**Date:** 2026-06-17

### What Was Built
- Full directory layout per Section 3 (`src/battery_rul/*`, `tests/`, `scripts/`, `notebooks/`, `data/`)
- `pyproject.toml` with runtime + dev dependencies, ruff/black/mypy/pytest config
- `src/battery_rul/config.py` with paths and baseline hyperparameters
- `.github/workflows/ci.yml` running ruff, black, mypy, pytest on push/PR to main
- `__init__.py` stubs for all `battery_rul` subpackages
- `.venv` created via `uv`, dependencies installed and verified importable

### Results / Metrics
- `pytest tests/` collects 0 tests (expected)
- `ruff check`, `black --check`, `mypy src/` all pass clean

### Issues Encountered
- None

### Next Phase
Phase 2 — Data Pipeline

## Phase 2 — Data Pipeline — COMPLETE

**Date:** 2026-06-17

### What Was Built
- `src/battery_rul/data/download.py` — downloads and extracts RW9-RW12 from the actual
  NASA PCoE archive, with SHA-256 checksums written to skip re-download on rerun
- `src/battery_rul/data/preprocess.py` — flattens each battery's nested step structure
  into a long DataFrame, computes SOH, adds rolling voltage stats / dv_dt / cycle_count,
  fits a MinMaxScaler on RW9 and applies it to all four batteries, writes parquet
- `src/battery_rul/data/dataset.py` — `BatteryDataset` (windowed feature/SOH pairs) and
  `get_dataloaders()`
- `scripts/download_data.py` CLI wrapper
- `tests/test_preprocess.py` — SOH formula, feature presence, scaler round-trip, dataset shapes

### Results / Metrics
- Full pipeline run on real data: RW9 8,532,073 / RW10 8,596,025 / RW11 8,664,510 /
  RW12 ~8.7M rows; train DataLoader yields 16,665 batches of shape (512, 50, 9)
- `pytest`, `ruff`, `black`, `mypy` all pass clean

### Issues Encountered
- CLAUDE.md assumed the dataset was served as flat CSVs via a Socrata API at
  `data.nasa.gov/Raw-Data/.../ugxu-9kjx`. That URL 404s — data.nasa.gov runs CKAN, not
  Socrata, and the actual files are MATLAB `.mat` structs (one per battery, each a list
  of ~113k "step" records with nested per-step arrays) inside a single zip at
  `data.nasa.gov/docs/legacy/ames/1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip`.
  `download.py` and `preprocess.py` were written against the real format.
- Applying the paper's literal SOH formula (`V0` = first voltage sample of the whole
  series) to the full random-walk history produces SOH outside [0, 1] (observed range on
  RW9: -0.36 to 2.07), since later reference/pulsed-charge steps spike above the first
  sample's instantaneous voltage. This is a property of the formula on raw data, not a
  bug — the unit test instead verifies the formula's correctness on a monotonic decay
  curve (V0 -> 0.8*V0), which is what the formula is designed for. Worth keeping in mind
  for dashboard SOH gauge clamping and training stability in later phases.

### Next Phase
Phase 3 — Models

## Phase 3 — Models — COMPLETE

**Date:** 2026-06-17

### What Was Built
- `src/battery_rul/models/dnn.py` — `PaperDNN` (exact replica: Linear->ReLU->Linear->
  Sigmoid->Linear, sized from `config.HIDDEN_LAYERS`) and `UpgradedDNN` (configurable
  depth/width, Linear->BatchNorm1d->ReLU->Dropout per layer, residual add when a block's
  input/output dims match)
- `src/battery_rul/models/lstm.py` — `BatteryLSTM` (2-layer LSTM, hidden=64, dropout=0.2,
  Linear(64,1) head over the final timestep)
- `tests/test_models.py` — forward-pass shapes for all three models, residual-path shape
  check, parameter-count sanity bounds

### Results / Metrics
- `pytest` (9 tests total), `ruff`, `black`, `mypy` all pass clean

### Issues Encountered
- Initial `UpgradedDNN` used `nn.ModuleDict` blocks inside an `nn.ModuleList`; mypy
  couldn't infer dict-style indexing on a generic `nn.Module`. Refactored to four
  parallel `nn.ModuleList`s (linears/norms/dropouts) iterated with `zip(..., strict=True)`
  — cleaner and fully typed.

### Next Phase
Phase 4 — Training, Tuning & Evaluation

## Phase 4 — Training, Tuning & Evaluation — COMPLETE

**Date:** 2026-06-17

### What Was Built
- `src/battery_rul/training/trainer.py` — `Trainer` (train/eval epoch loops, RMSE loss,
  early stopping with best-state restore, MLflow logging per run)
- `src/battery_rul/evaluation/metrics.py` — `rmse`, `mae`, `r2_score`,
  `overestimation_rmse`, `underestimation_rmse`, `safety_ratio`
- `src/battery_rul/training/tuning.py` — Optuna `objective`/`run_study` over `UpgradedDNN`
  (n_layers, per-layer width, lr, dropout, batch_size)
- `scripts/train.py` — CLI for all 4 required configs (`paper_dnn`/`upgraded_dnn`/`lstm`
  x approach 1/2), with a `stride` parameter on `BatteryDataset` to subsample windows
- `scripts/tune.py` — CLI wrapper around `run_study`, writes `data/processed/best_params.json`
- `tests/test_metrics.py` — 6 tests covering all metric functions

### Results / Metrics
Final results (paper_dnn/upgraded_dnn on full data, lstm and Optuna on stride-subsampled
data per the compute-scope decision below):

| Model | Approach | RW10 | RW11 | RW12 | Average RMSE |
|---|---|---|---|---|---|
| paper_dnn | 1 (intra-battery, RW9 80/20) | — | — | — | 0.18% (RW9 holdout) |
| paper_dnn | 2 (cross-battery) | 0.12% | 0.14% | 1.71% | **0.66%** |
| upgraded_dnn | 2 | 0.81% | 0.75% | 2.22% | **1.26%** |
| lstm | 2 (stride=5) | 0.31% | 0.41% | 1.76% | **0.83%** |

All three Approach-2 models beat the paper's reported 1.49% (Paper DNN), 1.55% (BLS-RVM),
and 1.61% (RNN+LSTM) baselines.

Optuna search (15 trials, stride=10 subsampled RW9, 10 epochs/trial, patience=3):
best val RMSE 0.88% with `{n_layers: 2, n_units_l0: 77, n_units_l1: 80, lr: 0.00135,
dropout: 0.015, batch_size: 256}`, saved to `data/processed/best_params.json`.

### Issues Encountered
- **Compute-scope decision (user-approved):** a full-spec run of all 4 training configs
  plus a 50-trial Optuna search was estimated at ~13-14 hours on this machine. User chose
  the reduced-scope option: `paper_dnn` and `upgraded_dnn` train on the full dataset;
  `lstm` (approach 2) uses `stride=5` window subsampling; Optuna uses `stride=10` and
  `n_trials=15` instead of 50. This is a compute-driven deviation from CLAUDE.md Section 6
  Phase 4, not a methodology change — `BatteryDataset` gained a `stride` parameter that
  only thins which windows are sampled, the windows themselves are unchanged.
- **Real bug found via first training pass, not assumed correct:** the initial Approach-2
  run (all 3 models) produced ~31-32% average RMSE across every model — same magnitude
  regardless of architecture, which pointed at the data pipeline rather than the models.
  Root cause: `compute_soh`'s `V0` was the first voltage sample of each battery's trace.
  Since a random-walk recording can start at an arbitrary point in the charge/discharge
  cycle, first-sample voltage varied widely across batteries (RW9=3.838V, RW10=4.196V,
  RW11=4.187V, RW12=3.957V), so the SOH target was a different affine function of voltage
  per battery — a model trained on RW9's mapping couldn't transfer to a different mapping.
  First fix attempt (`V0 = voltage.max()`) used each battery's true full-charge voltage
  instead (consistent ~4.65-4.81V across batteries) and dropped average RMSE to 0.66-4.94%
  range — but RW12 stayed an outlier (~13%) in every model. Investigated further: RW12's
  raw max (4.814V) was a single-sample sensor noise spike (next-highest value was 4.728V;
  99.9th percentile was 4.215V). Switched `V0` to `voltage.quantile(0.999)`, which recovers
  the same ~4.2V nominal full-charge voltage for all four batteries (the actual rated
  voltage for this 18650 cell) and is robust to single-sample noise. Re-ran preprocessing
  and all 4 training configs after each fix; final numbers above reflect the quantile-based
  `V0`. `tests/test_preprocess.py::test_soh_monotonic_decay_stays_in_unit_range` tolerances
  were loosened slightly (`abs=0.01`) since a quantile-based V0 can let the top ~0.1% of
  samples exceed SOH=1.0 by design.
- `mlflow.pytorch.log_model`'s newer default `serialization_format='pt2'` requires an
  `input_example` (traced-graph export); passed `serialization_format="pickle"` explicitly
  instead (acceptable security tradeoff for this local, non-served use case).

### Next Phase
Phase 5 — Interactive Dashboard

## Phase 5 — Interactive Dashboard — COMPLETE

**Date:** 2026-06-18

### What Was Built
- `src/battery_rul/data/preprocess.py` — `build_features()` now also keeps unscaled
  `absolute_time_raw`/`voltage_raw` columns (added after engineered features, before the
  MinMax scaler is fit/applied) so the dashboard can plot real seconds and volts instead
  of [0, 1]-scaled values; preprocessing re-run to regenerate all 4 parquets
- `src/battery_rul/evaluation/predictions.py` — `predict_battery()`, a windowed-inference
  helper that runs a trained model over a battery's processed parquet at a given stride
  and returns actual-vs-predicted SOH alongside the raw time/voltage columns
- `scripts/precompute_predictions.py` — loads each of the 4 Phase-4 models directly from
  MLflow (most recent run per run-name, via `MlflowClient.search_runs`) instead of
  retraining, runs `predict_battery` per battery at a stride chosen to cap each trace at
  ~5,000 points (browser-friendly), and pulls per-epoch `train_rmse`/`val_rmse` history
  via `get_metric_history` for the training-curve chart. Writes everything to
  `data/processed/predictions/*.parquet` (gitignored, same as other processed data)
- `src/battery_rul/dashboard/app.py` — Dash app: approach toggle (1/2) drives which
  models/batteries are selectable (Approach 1 only has `paper_dnn`/RW9; Approach 2 has
  all 3 models x RW10-12); voltage-vs-time trace, SOH gauge (red/yellow/green at the
  paper's 80/90% thresholds, clamped to [0, 100] for display), actual-vs-predicted SOH
  overlay, RMSE-per-epoch training curve, and a model comparison table (published
  baselines from `config.PAPER_BASELINE_RMSE` + our 3 models, best RMSE row bolded)
- `scripts/serve.py` — launches the dashboard at `http://localhost:8050`
- `src/battery_rul/dashboard/assets/style.css` — minimal sidebar/main-panel flex layout
- `docs/images/dashboard_screenshot.png` + a Dashboard section in `README.md`

### Results / Metrics
- RMSE computed from the precomputed (subsampled) prediction files matches the
  full-resolution Phase 4 numbers to within rounding (e.g. paper_dnn/RW12: 1.71% both
  ways), confirming the ~5,000-point subsample is representative for the table and charts
- `pytest` (15 tests), `ruff`, `black`, `mypy` all pass clean
- Manually exercised in a browser via Playwright: approach toggle correctly re-scopes the
  model/battery dropdowns, all 4 charts update on selection change, gauge color and
  comparison-table highlighting both verified visually

### Issues Encountered
- **Voltage-vs-predicted overlay isn't possible as literally specified**: CLAUDE.md's
  Phase 5 spec describes a "Voltage vs Time trace (actual + predicted overlay)," but every
  model in this project predicts SOH, not voltage — there is no predicted-voltage series to
  overlay. Adapted to the models' actual output: the voltage panel shows the real
  (unscaled) actual trace only, and the actual-vs-predicted overlay was moved to the SOH
  panel, which is the quantity the models actually predict.
- **SOH gauge clamping was needed, as flagged back in Phase 2**: the quantile-based SOH
  formula can produce values outside [0, 1] on noisy samples (confirmed live — RW12's
  raw predicted SOH hit -6.6% at one selected point). The gauge display is clamped to
  [0, 100]; the SOH-vs-time line chart is left unclamped so the chart still shows the true
  predicted value.
- Considered smoothing the gauge's "latest SOH" reading over the last 10 stride-subsampled
  points to reduce noise, but those points are spread across a wide time range at this
  stride (not actually "recent"), so averaging them would misrepresent the signal rather
  than clean it up. Kept the gauge as the single latest predicted point (matches what the
  model actually outputs at one instant), clamped for display only.
- Reused the already-trained Phase 4 MLflow models instead of retraining for prediction
  precomputation — `MlflowClient.search_runs` filtered by run name and sorted by start
  time picks up the most recent (post-SOH-fix) run automatically, so this works correctly
  even though each run name was trained 3 times across the Phase 4 bug-fix iterations.

### Next Phase
Phase 6 — FastAPI Inference Endpoint
