#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:11:05 2019

@author: morisset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from tmniai import manage_RM
#%%

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

f, axes = plt.subplots(2, 3, figsize=(14, 10))
for i in range(len(degrees)):
    ax = axes[0,i]

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    ax.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    ax.plot(X_test, true_fun(X_test), label="True function")
    ax.scatter(X, y, edgecolor='b', s=20, label="Samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim((0, 1))
    ax.set_ylim((-2, 2))
    ax.legend(loc="best")
    ax.set_title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))

hidden_layer_sizes_set = ( (3,), (10,5), (1000, 1000))
hidden_layer_sizes_strs = ('3', '10-5', '1000-1000')
for i in range(len(hidden_layer_sizes_set)):
    scaleit=True
    RM = manage_RM(RM_type='ANN', X_train=X, y_train=y, scaling=scaleit)
    RM.init_RM(hidden_layer_sizes=hidden_layer_sizes_set[i], 
               tol=1e-6, max_iter=1000, 
               activation='relu',
               solver='lbfgs')
    RM.train_RM()
    X_test = np.linspace(0, 1, 1000)
    RM.set_test(X_test, scaleit=scaleit)
    RM.predict(scoring=False)
    ax = axes[1,i]
    ax.plot(X_test, RM.pred, label="Model")
    ax.plot(X_test, true_fun(X_test), label="True function")
    ax.scatter(X, y, edgecolor='b', s=20, label="Samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim((0, 1))
    ax.set_ylim((-2, 2))
    ax.legend(loc="best")
    ax.set_title("ANN = {}".format(hidden_layer_sizes_strs[i]))

plt.show()