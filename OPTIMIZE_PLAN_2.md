# Keras training optimizations for AI4neb

Recommendations for the `K_ANN` regressor in `ai4neb/Regressor/RegressionModel.py`,
based on measurements made with `1.ComparePolynom.ipynb` (TF 2.18.1 / Keras 3.6.0, CPU,
30 training points, 1×2×1 tanh network). Nothing has been changed in AI4neb yet —
this file only lists what should be done.

## Measured baseline

The notebook's plain Keras cell (`optimizer='adam'`, `epochs=5000`, default batch size)
took ~183 s and still under-fitted. The same model, tuned as described below, reached a
better fit (RMS test ≈ 0.04) in ~3 s:

| Configuration | Time | RMS test |
|---|---|---|
| Adam lr=0.001, 5000 epochs (original) | ~183 s | poor |
| Adam lr=0.05, 500 full-batch epochs | ~15 s | 0.128 |
| Adam lr=0.05, 2000 steps via tiled data (see §2) | ~3 s | 0.040 |

The AI4neb `K_ANN` run in the same notebook (`epochs=15500`) took ~20 s per training and
would benefit from the same changes.

## 1. Expose and rethink the learning rate

**Problem.** Adam's default `learning_rate=0.001` is far too small for small networks /
smooth regression problems. It is the main reason huge epoch counts (5000–15500) are
needed to converge.

**What to do in AI4neb.**
- Add a `learning_rate` (or `lr`) kwarg to `init_RM` for `K_ANN`, passed to
  `keras.optimizers.Adam(learning_rate=...)` in the `model.compile` call
  (around `RegressionModel.py:374`).
- Keep 0.001 as the default for backward compatibility, but document that small
  networks train orders of magnitude faster with lr in the 0.01–0.05 range,
  optionally combined with a `ReduceLROnPlateau` callback for final accuracy.

## 2. Kill the per-epoch overhead for small training sets

**Problem.** `model.fit` has a fixed cost of ~35–40 ms **per epoch** (metric reset,
callback machinery), independent of the data size. When the training set fits in a
single batch, 1 epoch = 1 gradient step, so N epochs cost N × 40 ms of pure overhead:
the 3-minute training in the notebook was >99 % bookkeeping.

**What to do in AI4neb.** In `train_RM` (the `RM.fit(...)` call around
`RegressionModel.py:714`), when `n_samples / batch_size` is small (e.g. < 10 steps
per epoch), convert epochs into steps by tiling the training data:

```python
# k repetitions of the data -> k full-batch gradient steps per epoch
X_rep = np.tile(X_train, (k, 1))
y_rep = np.tile(y_train, (k, 1) if y_train.ndim > 1 else k)
model.fit(X_rep, y_rep, epochs=epochs // k, batch_size=n_samples,
          shuffle=False, verbose=0)
```

With `shuffle=False` and `batch_size=n_samples`, every batch is identical to the
original training set, so the result is mathematically the same as `epochs` plain
full-batch epochs — but the per-epoch overhead is paid `epochs/k` times instead of
`epochs` times. This is the change that took the notebook from 3 min to 3 s.
`k = max(1, min(100, epochs // 20))` is a reasonable heuristic. Only apply the trick
when the tiled array stays small (say < 10⁶ rows); large training sets do not suffer
from this overhead in the first place.

An equivalent, cleaner alternative is a custom `tf.function` training loop, but the
tiling trick keeps `model.fit`, the `history` object, and callbacks untouched.

## 3. Drop redundant metrics

**Problem.** Compiling with `loss='mse', metrics=['mse', 'mae']` (as in
`RegressionModel.py:374-376` and the notebook) recomputes the loss as a metric plus an
extra MAE at every step, adding overhead without new information.

**What to do in AI4neb.** Default to `metrics=None` when the loss is already `mse`
(the loss value is always available in `history.history['loss']`); keep `metrics` as a
user-overridable kwarg for those who want MAE.

## 4. Replace huge fixed epoch counts with early stopping

**Problem.** Defaults like `epochs=15500` train for a fixed wall-clock budget whether
or not the model converged after 1000 steps.

**What to do in AI4neb.** Add an optional early-stopping kwarg wired to

```python
keras.callbacks.EarlyStopping(monitor='loss', patience=100,
                              restore_best_weights=True)
```

so users can set a generous `epochs` ceiling and let training stop at convergence.
This mirrors the `tol` / `n_iter_no_change` behaviour of the sklearn `SK_ANN` backend,
making the two backends comparable.

## 5. Keras 3 model construction (`Input` instead of `input_dim`)

**Problem.** The first `Dense(..., input_dim=N_in)` (around `RegressionModel.py:340`
and `:420`) triggers under Keras 3:

> UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using
> Sequential models, prefer using an `Input(shape)` object as the first layer.

**What to do in AI4neb.** Start Sequential models with an explicit input layer:

```python
from keras.layers import Input
model = Sequential()
model.add(Input(shape=(N_in,)))
model.add(Dense(n_hidden, activation=activation))  # no input_dim
```

Identical model and parameter count; removes the warning and stays compatible with
future Keras versions. (Cosmetic side effect: `model.summary()` no longer lists a
separate input row.)

## Suggested priority

1. §1 learning rate — biggest effect on fit quality per epoch, trivial to add.
2. §2 tiling for small datasets — biggest effect on wall-clock time (60× here).
3. §5 `Input` layer — trivial, removes a user-visible warning.
4. §4 early stopping and §3 metrics — quality-of-life and minor speed-ups.
