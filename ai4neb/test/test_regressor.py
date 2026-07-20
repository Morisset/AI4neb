#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core test suite for ai4neb.manage_RM (scikit-learn and gradient-boosting backends).

These are real pytest tests with assertions (they fail loudly on a regression),
replacing the old print-only / try-except scaffolding. Keras-specific behaviour
lives in test_regressor_keras.py.
"""
import numpy as np
import pytest

from ai4neb import manage_RM
from ai4neb.Regressor.RegressionModel import XGB_OK, LGB_OK, CB_OK


# --------------------------------------------------------------------------- #
# Helpers / datasets
# --------------------------------------------------------------------------- #
def r2(y_true, y_pred):
    """R^2 coefficient, robust to (n,) / (n,1) shapes and multi-output."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return 1 - u / v


def make_data(seed=0, n=200):
    """A smooth, learnable regression problem (inputs -> outputs)."""
    rng = np.random.RandomState(seed)
    X1 = rng.uniform(0.0, 1.0, n)                     # 1 input
    y1 = np.sin(2 * np.pi * X1) + 0.05 * rng.randn(n)  # 1 output
    X2 = rng.uniform(0.0, 1.0, (n, 2))                # 2 inputs
    y2 = np.array([                                   # 2 outputs
        np.sin(2 * np.pi * X2[:, 0]) + 0.05 * rng.randn(n),
        np.cos(2 * np.pi * X2[:, 1]) + 0.05 * rng.randn(n),
    ]).T
    return X1, y1, X2, y2


@pytest.fixture(scope="module")
def data():
    return make_data()


# --------------------------------------------------------------------------- #
# Dimensions / shapes across the 4 input/output combinations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Xkey,ykey,exp_in,exp_out", [
    ("X1", "y1", 1, 1),
    ("X2", "y1", 2, 1),
    ("X1", "y2", 1, 2),
    ("X2", "y2", 2, 2),
])
def test_dims_and_pred_shape(data, Xkey, ykey, exp_in, exp_out):
    X1, y1, X2, y2 = data
    X = {"X1": X1, "X2": X2}[Xkey]
    y = {"y1": y1, "y2": y2}[ykey]

    RM = manage_RM(RM_type='SK_ANN', X_train=X, y_train=y,
                   split_ratio=0.3, random_seed=1)
    assert RM.N_in == exp_in
    assert RM.N_out == exp_out
    assert RM.N_train + RM.N_test == len(y)

    RM.init_RM(hidden_layer_sizes=(20,), max_iter=800, activation='tanh')
    RM.train_RM()
    RM.predict()
    pred = np.asarray(RM.pred)
    # one prediction row per test sample, regardless of the (n,)/(n,k) layout
    assert pred.shape[0] == RM.N_test
    assert np.isfinite(pred).all()


# --------------------------------------------------------------------------- #
# A numpy integer seed (e.g. from `4 + np.arange(5)`) is accepted
# (regression test: stdlib random.seed rejects numpy integer types)
# --------------------------------------------------------------------------- #
def test_numpy_integer_seed(data):
    X1, y1, _, _ = data
    for r_seed in 4 + np.arange(3):
        assert not isinstance(r_seed, int)          # it's a numpy scalar
        RM = manage_RM(RM_type='SK_ANN', X_train=X1, y_train=y1,
                       split_ratio=0.3, random_seed=r_seed)
        assert isinstance(RM.random_seed, int)      # normalised to native int
        RM.init_RM(hidden_layer_sizes=(10,), max_iter=200)
        RM.train_RM()                               # must not raise


# --------------------------------------------------------------------------- #
# The pipeline actually learns on strong backends
# --------------------------------------------------------------------------- #
def test_sk_ann_learns(data):
    X1, y1, _, _ = data
    RM = manage_RM(RM_type='SK_ANN', X_train=X1, y_train=y1,
                   split_ratio=0.3, scaling=True, random_seed=1)
    RM.init_RM(hidden_layer_sizes=(30, 30), max_iter=5000,
               activation='tanh', solver='lbfgs', tol=1e-8)
    RM.train_RM()
    assert RM.train_score[0] > 0.9          # scoring path used by train_RM works
    RM.predict()
    assert r2(RM.y_test, RM.pred) > 0.8      # generalises to held-out points


def test_poly_fits_polynomial_data():
    """A degree-3 Poly model recovers a cubic almost exactly."""
    rng = np.random.RandomState(3)
    X = rng.uniform(-2, 2, 200)
    y = 1.0 + 2.0 * X - 0.5 * X ** 2 + 0.3 * X ** 3
    RM = manage_RM(RM_type='Poly', X_train=X, y_train=y,
                   split_ratio=0.3, random_seed=1)
    RM.init_RM(degree=3)
    RM.train_RM()
    RM.predict()
    assert r2(RM.y_test, RM.pred) > 0.999


@pytest.mark.parametrize("RM_type", ['SK_SVM', 'SK_GBR', 'SK_RFR'])
def test_other_sklearn_backends_learn(data, RM_type):
    X1, y1, _, _ = data
    RM = manage_RM(RM_type=RM_type, X_train=X1, y_train=y1,
                   split_ratio=0.3, scaling=True, random_seed=1)
    RM.init_RM()
    RM.train_RM()
    RM.predict()
    assert np.asarray(RM.pred).shape[0] == RM.N_test
    assert r2(RM.y_test, RM.pred) > 0.7


# --------------------------------------------------------------------------- #
# predict(scoring=True) returns a real R^2 for multi_predic backends
# (regression test: this path used to always yield nan)
# --------------------------------------------------------------------------- #
def test_predict_scoring_single_output(data):
    X1, y1, _, _ = data
    RM = manage_RM(RM_type='SK_ANN', X_train=X1, y_train=y1,
                   split_ratio=0.3, scaling=True, random_seed=1)
    RM.init_RM(hidden_layer_sizes=(30, 30), max_iter=4000,
               activation='tanh', solver='lbfgs')
    RM.train_RM()
    RM.predict(scoring=True)
    s = np.asarray(RM.predic_score)
    assert s.ndim == 0                      # a scalar for a single output
    assert np.isfinite(s)
    assert s > 0.8
    # matches an independent R^2 on the stored predictions
    assert float(s) == pytest.approx(r2(RM.y_test, RM.pred), abs=1e-6)


def test_predict_scoring_multi_output(data):
    _, _, X2, y2 = data
    RM = manage_RM(RM_type='SK_ANN', X_train=X2, y_train=y2,
                   split_ratio=0.3, scaling=True, random_seed=1)
    RM.init_RM(hidden_layer_sizes=(30, 30), max_iter=4000,
               activation='tanh', solver='lbfgs')
    RM.train_RM()
    RM.predict(scoring=True)
    s = np.asarray(RM.predic_score)
    assert s.shape == (RM.N_out,)           # one score per output
    assert np.isfinite(s).all()
    # each entry is the per-output R^2
    pred = np.asarray(RM.pred)
    yt = np.asarray(RM.y_test)
    for j in range(RM.N_out):
        u = ((yt[:, j] - pred[:, j]) ** 2).sum()
        v = ((yt[:, j] - yt[:, j].mean()) ** 2).sum()
        assert float(s[j]) == pytest.approx(1 - u / v, abs=1e-6)


# --------------------------------------------------------------------------- #
# Discretisation + reduce_by
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reduce_by", ['mean', 'mean_norm', 'max'])
def test_discretization_reduce(data, reduce_by):
    X1, y1, _, _ = data
    RM = manage_RM(RM_type='SK_ANN', X_train=X1, y_train=y1,
                   split_ratio=0.3, N_y_bins=41, scaling=True, random_seed=1)
    assert RM.discretized
    RM.init_RM(hidden_layer_sizes=(30,), max_iter=1500)
    RM.train_RM()
    RM.predict(reduce_by=reduce_by)
    pred = np.asarray(RM.pred)
    assert pred.shape[0] == RM.N_test
    assert np.isfinite(pred).all()
    # 'mean_norm' and 'max' are convex combinations / grid picks and so stay
    # inside the discretisation grid; plain 'mean' uses un-normalised weights
    # and may fall outside it, so it is only checked for finiteness above.
    if reduce_by in ('mean_norm', 'max'):
        assert pred.min() >= RM.y_vects.min() - 1e-9
        assert pred.max() <= RM.y_vects.max() + 1e-9


# --------------------------------------------------------------------------- #
# Save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_sklearn(data, tmp_path):
    X1, y1, _, _ = data
    fn = str(tmp_path / "rm_sk")
    RM = manage_RM(RM_type='SK_ANN', X_train=X1, y_train=y1,
                   scaling=True, random_seed=1)
    RM.init_RM(hidden_layer_sizes=(20,), max_iter=1000)
    RM.train_RM()
    RM.set_test(X1)
    RM.predict()
    pred_before = np.asarray(RM.pred).ravel().copy()

    RM.save_RM(filename=fn)

    RM2 = manage_RM(RM_filename=fn)
    assert RM2.model_read
    assert RM2.RM_type == 'SK_ANN'
    RM2.set_test(X1)
    RM2.predict()
    pred_after = np.asarray(RM2.pred).ravel()
    np.testing.assert_allclose(pred_before, pred_after, rtol=0, atol=0)


def test_save_test_roundtrip_unscaled(data, tmp_path):
    """save_test=True with scaling=False used to crash (missing y_test_unscaled)."""
    X1, y1, _, _ = data
    fn = str(tmp_path / "rm_unscaled")
    RM = manage_RM(RM_type='SK_ANN', X_train=X1, y_train=y1,
                   X_test=X1, y_test=y1, scaling=False, random_seed=1)
    RM.init_RM(hidden_layer_sizes=(20,), max_iter=800)
    RM.train_RM()
    RM.predict()
    pred_before = np.asarray(RM.pred).ravel().copy()

    RM.save_RM(filename=fn, save_train=True, save_test=True)   # must not raise

    RM2 = manage_RM(RM_filename=fn)
    assert RM2.model_read
    # the saved (unscaled) test set survives the round-trip
    assert RM2.X_test is not None and RM2.y_test is not None
    RM2.predict()
    np.testing.assert_allclose(pred_before, np.asarray(RM2.pred).ravel(),
                               rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Optional gradient-boosting backends (skipped when the lib is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not XGB_OK, reason="xgboost not installed")
def test_xgb_learns(data):
    X1, y1, _, _ = data
    RM = manage_RM(RM_type='XGB', X_train=X1, y_train=y1,
                   split_ratio=0.3, random_seed=1)
    RM.init_RM(n_estimators=200, max_depth=3)
    RM.train_RM()
    RM.predict()
    assert r2(RM.y_test, RM.pred) > 0.7


@pytest.mark.skipif(not LGB_OK, reason="lightgbm not installed")
def test_lgb_learns(data):
    X1, y1, _, _ = data
    RM = manage_RM(RM_type='LGB', X_train=X1, y_train=y1,
                   split_ratio=0.3, random_seed=1)
    RM.init_RM(n_estimators=200)
    RM.train_RM()
    RM.predict()
    assert r2(RM.y_test, RM.pred) > 0.7


@pytest.mark.skipif(not CB_OK, reason="catboost not installed")
def test_catboost_learns_and_roundtrips(data, tmp_path):
    X1, y1, _, _ = data
    fn = str(tmp_path / "rm_cb")
    RM = manage_RM(RM_type='CatBoost', X_train=X1, y_train=y1,
                   X_test=X1, y_test=y1, random_seed=1)
    RM.init_RM(iterations=200, depth=3, verbose=False)
    RM.train_RM()
    RM.predict()
    assert r2(y1, RM.pred) > 0.7
    pred_before = np.asarray(RM.pred).ravel().copy()

    # round-trip: the loader must rebuild CatBoost models (not XGBoost)
    RM.save_RM(filename=fn)
    RM2 = manage_RM(RM_filename=fn)
    assert RM2.model_read
    RM2.set_test(X1)
    RM2.predict()
    np.testing.assert_allclose(pred_before, np.asarray(RM2.pred).ravel(),
                               rtol=1e-5, atol=1e-5)
