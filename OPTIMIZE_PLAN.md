# Plan: Optimize RegressionModel.py

## Context

`ai4neb/Regressor/RegressionModel.py` is the core module of the `ai4neb` package (1,139 lines). It wraps scikit-learn, Keras/TF, XGBoost, LightGBM, and CatBoost regressors under a unified `manage_RM` class. The file has accumulated several performance inefficiencies and two latent bugs.

---

## Changes

### 1. `_discretize1()` — lines 540–544

**Two `np.where()` → `np.clip()`** (one array scan instead of two):
```python
# Before
tt = np.where(tt < 0, 0, tt)
tt = np.where(tt > self.N_y_bins[i]-1, self.N_y_bins[i]-1, tt)

# After
tt = np.clip(tt, 0, self.N_y_bins[i] - 1)
```

**Precompute cumsum outside the loop** (currently recomputed every iteration):
```python
# Before (inside loop)
if i > 0:
    tt += np.cumsum(self.N_y_bins)[i-1]

# After
cumsum_bins = np.cumsum(self.N_y_bins)   # before loop
...
if i > 0:
    tt += cumsum_bins[i-1]
```

### 2. `_norm_pred()` — lines 767–768

Replace `np.expand_dims()` calls with `keepdims=True`:
```python
# Before
tmp = self.pred - np.expand_dims(self.pred.min(1), axis=1)
self.pred_norm = tmp / np.expand_dims(tmp.sum(1), axis=1)

# After
tmp = self.pred - self.pred.min(1, keepdims=True)
self.pred_norm = tmp / tmp.sum(1, keepdims=True)
```

### 3. `predict()` reduce section — lines 862–869

Precompute `cumsum` once outside the loop (currently called twice per iteration):
```python
# Before (inside loop)
i_inf = self.N_y_bins.cumsum()[i-1]
i_sup = self.N_y_bins.cumsum()[i]

# After
cumsum_bins = self.N_y_bins.cumsum()   # before loop
...
i_inf = 0 if i == 0 else cumsum_bins[i-1]
i_sup = cumsum_bins[i]
```

### 4. Bug fix: wrong print message — line 876

```python
# Before
print('Reducing y by mean')   # wrong: copy-paste error

# After
print('Reducing y by normalized mean')
```

### 5. Bug fix: CatBoost load uses wrong class — around line 1091

The `load_RM` method creates an `xgb.XGBRegressor()` placeholder when loading a CatBoost model. Fix:
```python
# Before
model = xgb.XGBRegressor()

# After
model = catboost.CatBoostRegressor()
```

---

## Files Modified

- `ai4neb/Regressor/RegressionModel.py` — all changes above

## Verification

Run the existing test suite after changes:
```bash
python -m pytest ai4neb/test/test_regressor.py -v
```

Also manually verify discretization round-trips produce identical results before and after the `np.clip` / cumsum refactor.
