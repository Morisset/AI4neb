# AI4neb session report — Keras optimizations, bug fixes, test suite & packaging

This document reports the changes made to `ai4neb` in this working session. Unlike
`OPTIMIZE_PLAN.md` and `OPTIMIZE_PLAN_2.md` (which were *plans* describing work to be
done), this file records what was **actually implemented, verified, and committed**.

All work landed on branch `fabel_optmization` in the commit range `14a12f5..HEAD`.
The full test suite (`ai4neb/test/`) passes: **33 passed**.

---

## 1. Keras (`K_ANN`) training optimizations

Implementation of the five recommendations from `OPTIMIZE_PLAN_2.md`, plus two default
changes so the backend is fast/accurate out of the box. All changes are in
`ai4neb/Regressor/RegressionModel.py`, in the `init_RM` / `train_RM` methods of
`manage_RM`. They are backward-compatible for the trained model and its predictions;
the exceptions (metrics keys, loss-curve length) are noted below.

Measured effect (30-point, 1→2→1 tanh network from `docs/ComparePolynom.ipynb`,
TF 2.18.1 / Keras 3.6.0, CPU): the original AI4neb `K_ANN` run (`epochs=15500`, default
`lr=0.001`) took ~21 s and under-fit; the same call now runs materially faster, and with
the optimized defaults a comparable fit is reached in a few seconds.

### 1.1 Learning-rate kwarg (§1)
- Added a `learning_rate` (alias `lr`) kwarg. When the optimizer is the default string
  `'adam'`, it is now built as an explicit `optimizers.Adam(learning_rate=...)`.
- A user-supplied optimizer object or a different solver string is left untouched.

### 1.2 Tiling small training sets (§2)
- New helpers `manage_RM._tile_for_keras()` and `manage_RM._keras_fit()`.
- `model.fit` has a fixed per-epoch overhead (~35–40 ms) independent of data size. When
  a small training set trains in a single/few full-batch steps, that overhead dominates.
  `_tile_for_keras` tiles the data `k` times and divides `epochs` by `k`, so the overhead
  is paid `epochs/k` times instead of `epochs` times. With `shuffle=False` and an unchanged
  `batch_size`, every tile reproduces the exact same sequence of batches, so the result is
  **mathematically equivalent** to plain full-batch training — only bookkeeping is saved.
- Heuristic: `k = max(1, min(100, epochs // 20))`. Tiling is **skipped** when the model
  already runs ≥ 10 steps/epoch, when a `validation_split` is set (identical rows would
  leak across the split), or when the tiled array would exceed ~10⁶ rows.
- Applied only for `RM_type in ('K_ANN', 'K_ANN_Dis')`; every non-Keras backend calls
  `RM.fit` exactly as before. The phased (`tuple`/`list` `batch_size`/`epochs`) training
  path still works, with tiling applied per phase.
- Caveat: when tiling triggers, `epochs` is reduced internally, so
  `history.history['loss']` (and therefore `plot_loss`) shows a shorter/coarser curve for
  small datasets. The final loss and model are equivalent.

### 1.3 Drop redundant default metrics (§3)
- `K_ANN` default changed from `metrics=['mse','mae']` to `metrics=None`. The loss is
  already `mse`, so `history.history['loss']` still carries it; the extra metric passes are
  removed. `K_ANN_Dis` (`metrics=['accuracy']`, cross-entropy loss) is untouched.
- Caveat: callers that read `history.history['mse']` / `['mae']` must now pass `metrics`
  explicitly.

### 1.4 Optional early stopping (§4)
- Added an `early_stopping` kwarg wired to
  `callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)`.
- Accepts `True`/`False` or a dict of `EarlyStopping` overrides (e.g. `{'patience': 7}`).

### 1.5 Keras 3 `Input` layer instead of `input_dim` (§5)
- Both Sequential builders (`K_ANN`/`K_ANN_Dis` and the inner `create_model` of
  `KSK_ANN`) now start with `Input(shape=(N_in,))` and drop `input_dim=`. Identical model
  and parameter count; removes the Keras 3 `UserWarning` and is future-proof.
- Imports updated in all three keras-access fallback blocks: added `Input` to the layers
  import and `optimizers`, `callbacks` to the top-level keras import.
- Follow-up fix: the first `Dropout` layer still passed `input_shape=(...)`, so building a
  `K_ANN` **with dropout** re-triggered the same warning. Removed that argument (redundant —
  the Dropout follows the first Dense, whose output shape is known). The warning regression
  test now also covers the dropout path.

### 1.6 Optimized defaults (out-of-the-box tuning)
- `learning_rate` default: **0.01** (was Adam's stock 0.001). 10× faster convergence for
  the small/smooth networks this package targets; chose the conservative end of the
  0.01–0.05 range because `manage_RM`'s own default `hidden_layer_sizes=(10,10)` is larger
  than the notebook toy net, where 0.05 risks instability.
- `early_stopping` default: **True**. Callers can restore the old behaviour with
  `early_stopping=False`.
- Note: `patience=100` counts *tiled* epochs, so on small datasets early stopping is a
  soft convergence guard (each tiled epoch is many real gradient steps), not a hard cap.

### 1.7 Usage notes / gotchas observed in practice
- **`validation_split` disables tiling.** When `validation_split > 0`, `_tile_for_keras`
  returns the data unchanged (duplicated rows would leak across Keras's end-of-array
  validation cut), so the run trains the full `epochs` with no speedup. To get tiling's
  speed, hold out test data via `split_ratio` instead and leave `validation_split=0`.
- **Default early stopping monitors training `loss`.** With `dropout` and/or a
  `validation_split`, monitoring the (noisy) training loss is usually not what you want —
  pass `early_stopping={'monitor': 'val_loss'}` so it stops on generalization and
  `restore_best_weights` restores the best *validation* epoch.
- **Small nets at low epoch counts need a higher `lr`.** With the default `lr=0.01`, a
  tiny net (e.g. `(2,)`) needs enough gradient steps to converge: `epochs=2000` under-fits
  (RMS ≈ 0.13) whereas `epochs=15500` converges (RMS ≈ 0.039). Raising `learning_rate` to
  0.05 makes `epochs=2000` converge (RMS ≈ 0.039) — matching the notebook's raw-TF cell.

---

## 2. Bug fixes in `manage_RM`

### 2.1 `predict(scoring=True)` returned `nan` for every `_multi_predic` backend
Affected `SK_ANN`, `K_ANN`, `SK_RFR`, `SK_GPR`. Two root causes:
- The scoring call referenced `to_predict`, a name only bound in the *non*-multi branch, so
  it raised `NameError`, which the bare `except` swallowed into `nan`. Fixed to use
  `self.X_test` (the actual model input).
- The module-level `score()` then mis-broadcast a `(N,1)` `y_test` against a `(N,)`
  prediction, and used a single global mean for multi-output. Fixed `score()` to ravel a
  `(N,1)` `y_true` and use an axis-aware mean, so scoring returns a **scalar R²** for a
  single output and a **correct per-output R² array** for multi-output.
- Behaviour note: the multi-output path now uses per-output means (standard R²). Since the
  old code only ever produced `nan` on this path, no caller depended on the old numbers.

### 2.2 `save_RM(save_test=True)` crashed with `scaling=False`
`AttributeError: y_test_unscaled`. The scaling path sets `y_test_unscaled` via
`scale_sets`, but the three `scaling=False` branches (`__init__`, `set_test`, `load_RM`)
each set the other `*_unscaled` attributes and omitted this one. Added the missing
assignment to all three, so the unscaled test set is available to save and round-trips.

---

## 3. Test suite rewrite

The old `ai4neb/test/test_regressor.py` had no real assertions (everything was wrapped in
`try/except` + `print`, so it "passed" by never checking anything) and five outright-broken
tests calling the API with arguments/values that no longer exist (`noise=`,
`RM_type='Keras'`/`'KerasDis'`/`'ANN'`, and a TF1-era `tf.random.set_random_seed` path).
It was replaced with two assertion-based pytest files.

### `ai4neb/test/test_regressor.py` — core backends
- Dimension/shape correctness across all four input/output combinations.
- Real learning checks (R² thresholds) for `SK_ANN`, `Poly` (near-exact cubic recovery),
  `SK_SVM`/`SK_GBR`/`SK_RFR`, and `XGB`/`LGB`/`CatBoost` (each `skipif`-guarded on
  availability).
- Discretisation with all three `reduce_by` modes (`mean`, `mean_norm`, `max`); the
  grid-range bound is asserted only for the two that are guaranteed bounded — plain `mean`
  uses un-normalised weights and can legitimately fall outside the grid.
- Save/load round-trips for `SK_ANN` and `CatBoost` (the latter guards the CatBoost-loader
  fix from the earlier optimization pass).
- Regression tests for the §2 bug fixes: `test_predict_scoring_single_output`,
  `test_predict_scoring_multi_output`, `test_save_test_roundtrip_unscaled`.

### `ai4neb/test/test_regressor_keras.py` — Keras backend & optimizations
Module-skipped when TensorFlow is unavailable. Covers: end-to-end `K_ANN` training and
quality on defaults; the `learning_rate`/`lr` kwarg and the optimized `0.01` default;
metrics dropped (`history` has only `loss`); early stopping on by default, disablable, and
override-accepting; no `input_dim` deprecation warning; the `_tile_for_keras` transform
(expands small sets, and is skipped for large step-counts and with a `validation_split`);
and a `K_ANN` save/load round-trip.

Testing notes: quality is asserted via `train_score` and a manually computed R² rather than
solely `predic_score`, and the datasets/solvers were chosen (e.g. `lbfgs` for the tiny
smooth `SK_ANN` fit) so thresholds are stable across seeds.

---

## 4. Repository hygiene & packaging

### 4.1 Stop tracking `catboost_info/`
CatBoost writes `catboost_info/` (training logs, tfevents, tsv metrics) to the working
directory on every fit, so these files reappeared as "modified" after each test/model run.
Removed from version control (`git rm -r --cached`) and added `catboost_info/` to
`.gitignore`.

### 4.2 Remove unused `version.py`
`ai4neb/version.py` was dead code: nothing imported it, it did not back
`ai4neb.__version__`, and its value (`0.2.15b2`) was stale relative to `pyproject.toml`
(`0.3.2`). Deleted.

### 4.3 `ai4neb.__version__` now works and tracks `pyproject.toml` live
Several notebooks (`SaveRestore`, `Test_forward`, `OxygenDiags2`) print
`ai4neb.__version__`, but the package never defined it, so those cells raised
`AttributeError`. It is now computed in `ai4neb/__init__.py`:
- Prefer reading `[project].version` straight from `pyproject.toml` when it sits next to
  the loaded package (source tree / editable install), guarded by a `name == "ai4neb"`
  check — so a version bump is reflected **immediately, without reinstalling** (the
  previous `importlib.metadata`-only version returned the stale install-time snapshot).
- Fall back to `importlib.metadata.version("ai4neb")`, then `"0.0.0"`, for wheel installs
  where `pyproject.toml` is not shipped.
- `tomllib` is stdlib on Python 3.11+, with a `tomli` backport attempt for 3.8–3.10.

Verified: reports the live `0.3.2`, and a temporary bump to `9.9.9` was reflected without
reinstall.

---

## 5. Commits (this session)

| Commit | Summary |
|---|---|
| `5197211` | apply OPTIMIZE_PLAN_2 to speed up Keras small-size training |
| `3cd7060` | use optimized default values for learning rate and early stop |
| `ef030da` | debug and add tests (rewrite `test_regressor.py`, add `test_regressor_keras.py`) |
| `71e6c53` | fix predict scoring for multi_predic backends and unscaled `save_test` |
| `795f82d` | stop tracking `catboost_info/` training artifacts |
| `e30bfb4` | remove unused `version.py` |
| `0ff94a6` | expose `ai4neb.__version__` from installed package metadata |
| `1d1f3e5` | report live `pyproject.toml` version from a source/editable checkout |
| `5756be7` | drop `input_shape` from the first Dropout layer (Keras 3 warning) |

## 6. Verification

- `python -m pytest ai4neb/test/ -q` → **33 passed**.
- Notebook `K_ANN` scenario (`docs/ComparePolynom.ipynb`) reproduced with default kwargs:
  same fit quality (RMS test ≈ 0.039), faster wall-clock; with the optimized defaults the
  fit is reached in a few seconds.
- `import ai4neb; ai4neb.__version__` → `0.3.2` (live from `pyproject.toml`).

## 7. Known pre-existing issues left untouched (out of scope)

- The `KSK_ANN` backend still calls `tf.random.set_random_seed` (a removed TF1 API) and
  errors under current TensorFlow. `KSK_ANN` also depends on `scikeras`, which is not
  installed in the reference environment.
- Some `docs/` notebooks still call the API with the removed `noise=` kwarg and legacy
  `RM_type` strings (`'Keras'`, `'KerasDis'`, `'ANN'`); those cells need updating to the
  current API (`'K_ANN'`, `'K_ANN_Dis'`, `'SK_ANN'`, …).
