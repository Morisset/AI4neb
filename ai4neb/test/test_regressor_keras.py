#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Keras (K_ANN) backend of ai4neb.manage_RM, including the training
optimizations added from OPTIMIZE_PLAN_2.md:

  * a `learning_rate` / `lr` kwarg (default 0.01) wired into the Adam optimizer,
  * automatic tiling of small training sets to amortize per-epoch overhead,
  * `metrics=None` by default for K_ANN (loss already carries the mse),
  * early stopping on by default,
  * an explicit `Input` layer instead of the deprecated `input_dim` argument.

The whole module is skipped when TensorFlow/Keras is unavailable.
"""
import warnings

import numpy as np
import pytest

from ai4neb import manage_RM
from ai4neb.Regressor.RegressionModel import TF_OK

pytestmark = pytest.mark.skipif(not TF_OK, reason="TensorFlow/Keras not installed")


def r2(y_true, y_pred):
    y_true = np.ravel(np.asarray(y_true))
    y_pred = np.ravel(np.asarray(y_pred))
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - u / v


def make_data(seed=0, n=60):
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(0.0, 1.0, n))
    y = np.cos(1.5 * np.pi * X) + 0.05 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def data():
    return make_data()


# --------------------------------------------------------------------------- #
# End-to-end: a small K_ANN trains and fits well, fast, on defaults
# --------------------------------------------------------------------------- #
def test_kann_learns_on_defaults(data):
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y,
                   scaling=False, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(10,), epochs=2000, activation='tanh')
    RM.train_RM()
    assert RM.train_score[0] > 0.9
    RM.set_test(X)
    RM.predict()
    assert np.asarray(RM.pred).shape[0] == len(X)
    assert r2(y, RM.pred) > 0.9


# --------------------------------------------------------------------------- #
# §3 - redundant default metrics dropped (history carries only 'loss')
# --------------------------------------------------------------------------- #
def test_kann_default_metrics_dropped(data):
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(5,), epochs=100)
    RM.train_RM()
    assert list(RM.history[0].history.keys()) == ['loss']


# --------------------------------------------------------------------------- #
# §1 - learning_rate / lr kwarg drives the Adam optimizer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", ["learning_rate", "lr"])
def test_kann_learning_rate_kwarg(data, kw):
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(5,), epochs=1, **{kw: 0.033})
    opt = RM.RMs[0].optimizer
    assert type(opt).__name__ == 'Adam'
    assert float(np.array(opt.learning_rate)) == pytest.approx(0.033)


def test_kann_default_learning_rate_is_optimized(data):
    """Default lr is the optimized 0.01, not Adam's stock 0.001."""
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(5,), epochs=1)
    opt = RM.RMs[0].optimizer
    assert float(np.array(opt.learning_rate)) == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# §4 - early stopping on by default, and switchable
# --------------------------------------------------------------------------- #
def test_kann_early_stopping_on_by_default(data):
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(5,), epochs=100)
    cbs = RM.train_params.get('callbacks', [])
    assert any(type(c).__name__ == 'EarlyStopping' for c in cbs)


def test_kann_early_stopping_can_be_disabled(data):
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(5,), epochs=100, early_stopping=False)
    assert 'callbacks' not in RM.train_params


def test_kann_early_stopping_accepts_overrides(data):
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(5,), epochs=100,
               early_stopping={'patience': 7})
    es = RM.train_params['callbacks'][0]
    assert type(es).__name__ == 'EarlyStopping'
    assert es.patience == 7


# --------------------------------------------------------------------------- #
# §5 - explicit Input layer: no input_dim/input_shape deprecation warning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dropout", [None, (0.2, 0.3)])
def test_kann_no_input_dim_warning(data, dropout):
    # the first Dropout layer used to carry input_shape=..., which re-triggered
    # the Keras 3 warning whenever dropout was enabled
    X, y = data
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RM.init_RM(hidden_layer_sizes=(6, 4), epochs=1, dropout=dropout)
    msgs = " ".join(str(w.message) for w in caught)
    assert "input_dim" not in msgs and "input_shape" not in msgs


# --------------------------------------------------------------------------- #
# §2 - tiling transform for small training sets (unit test of the helper)
# --------------------------------------------------------------------------- #
def _kann_instance():
    X = np.linspace(0, 1, 30)
    y = np.cos(X)
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y, random_seed=0)
    RM.init_RM(hidden_layer_sizes=(4,), epochs=2000)
    return RM


def test_tiling_expands_small_dataset():
    RM = _kann_instance()
    X = np.arange(30, dtype=float)[:, None]
    y = np.arange(30, dtype=float)
    params = {'epochs': 2000, 'batch_size': None, 'validation_split': 0.0}
    X_rep, y_rep, new_params = RM._tile_for_keras(X, y, params)

    k = max(1, min(100, 2000 // 20))            # == 100
    assert X_rep.shape[0] == 30 * k
    assert y_rep.shape[0] == 30 * k
    assert new_params['epochs'] == 2000 // k    # 20 meta-epochs
    assert new_params['shuffle'] is False
    assert new_params['batch_size'] == 30       # full-batch
    # every tile is an exact copy -> equivalent to plain full-batch training
    np.testing.assert_array_equal(X_rep[:30], X_rep[30:60])
    np.testing.assert_array_equal(y_rep[:30], y_rep[30:60])


def test_tiling_skipped_for_large_step_counts():
    RM = _kann_instance()
    X = np.zeros((1000, 1))
    y = np.zeros(1000)
    params = {'epochs': 500, 'batch_size': 32, 'validation_split': 0.0}
    X_rep, y_rep, new_params = RM._tile_for_keras(X, y, params)
    # >= 10 steps/epoch already: overhead is not the bottleneck, leave untouched
    assert X_rep.shape[0] == 1000
    assert new_params == params


def test_tiling_skipped_with_validation_split():
    RM = _kann_instance()
    X = np.zeros((30, 1))
    y = np.zeros(30)
    params = {'epochs': 2000, 'batch_size': None, 'validation_split': 0.2}
    X_rep, y_rep, new_params = RM._tile_for_keras(X, y, params)
    # tiling would corrupt a validation split (identical rows leak across it)
    assert X_rep.shape[0] == 30
    assert new_params == params


# --------------------------------------------------------------------------- #
# Save / load round-trip for the Keras backend
# --------------------------------------------------------------------------- #
def test_kann_save_load_roundtrip(data, tmp_path):
    X, y = data
    fn = str(tmp_path / "rm_k")
    RM = manage_RM(RM_type='K_ANN', X_train=X, y_train=y,
                   scaling=True, random_seed=10)
    RM.init_RM(hidden_layer_sizes=(10,), epochs=500, activation='tanh')
    RM.train_RM()
    RM.set_test(X)
    RM.predict()
    pred_before = np.asarray(RM.pred).ravel().copy()

    RM.save_RM(filename=fn)

    RM2 = manage_RM(RM_filename=fn)
    assert RM2.model_read
    assert RM2.RM_type == 'K_ANN'
    RM2.set_test(X)
    RM2.predict()
    np.testing.assert_allclose(pred_before, np.asarray(RM2.pred).ravel(),
                               rtol=1e-5, atol=1e-5)
