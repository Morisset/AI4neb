#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:01:37 2019

@author: morisset
"""
import numpy as np
from mwinai import manage_RM
#%%

def test(func = 'sins'):
    n_samples=200
    if func == 'sins':
        X1 = np.random.randn(n_samples)
        y1 = np.sin(X1) + np.cos(X1) + 0.1*np.random.randn(n_samples)
        X2 = np.random.randn(n_samples, 2)
        y2 = np.array([np.sin(X2[:,0]) + np.cos(X2[:,1]) + 0.1*np.random.randn(n_samples),
             np.sin(X2[:,0])**2 + np.cos(X2[:,1])**2 + 0.1*np.random.randn(n_samples)]).T
    elif func == 'x24':
        y1 = np.random.uniform(-5, 5, n_samples)
        X1 = y1**2
        y2 = np.random.uniform(-5, 5, (n_samples, 2))
        X2 = np.array([y1**2, y1**4]).T
        
    
    print('X1 shape:', X1.shape)
    print('X2 shape:', X2.shape)
    print('y1 shape:', y1.shape)
    print('y2 shape:', y2.shape)
    
    N_y_bins = None
    y_vects = np.linspace(-5,5,41)
    if False:
        N_y_bins = 41
        y_vects = None
    
    scoring = True
    reduce_by = None#'mean'
    split_ratio=0.3
    verbose = False
    try:
        print('-----')
        print
        RM1 = manage_RM(X_train=X1, y_train=y1, split_ratio=split_ratio, verbose=verbose, 
                        N_y_bins=N_y_bins, y_vects=y_vects, RM_type='ANN')
        RM1.init_RM(max_iter=200000, tol=0.00001, solver='lbfgs', activation='tanh',
                    hidden_layer_sizes=(10))
        RM1.train_RM()
        RM1.predict(scoring=scoring, reduce_by=reduce_by)
        print(RM1.N_train, RM1.N_in, RM1.N_train_y, RM1.N_out)
    except:
        print("!!! TEST 1 not passed !!!")
     
    if True:
        
        try:
            print('-----')
            print
            RM2 = manage_RM(X_train=X2, y_train=y1, split_ratio=split_ratio, verbose=verbose, 
                            N_y_bins=N_y_bins, y_vects=y_vects)
            RM2.init_RM(max_iter=200, tol=0.001)
            RM2.train_RM()
            RM2.predict(scoring=scoring, reduce_by=reduce_by)
            print(RM2.N_train, RM2.N_in, RM2.N_train_y, RM2.N_out)
        except:
            print("!!! TEST 2 not passed !!!")
        y_vects = [np.linspace(-5,5,41), np.linspace(-6,6,41)]
        try:
            print('-----')
            print
            RM3 = manage_RM(X_train=X1, y_train=y2, split_ratio=split_ratio, verbose=verbose, 
                            N_y_bins=N_y_bins, y_vects=y_vects)
            RM3.init_RM(max_iter=200, tol=0.001)
            RM3.train_RM()
            RM3.predict(scoring=scoring, reduce_by=reduce_by)
            print(RM3.N_train, RM3.N_in, RM3.N_train_y, RM3.N_out)
        except:
            print("!!! TEST 3 not passed !!!")
            RM3 = manage_RM(X_train=X1, y_train=y2, split_ratio=split_ratio, verbose=verbose, 
                            N_y_bins=N_y_bins, y_vects=y_vects)
            RM3.init_RM(max_iter=200, tol=0.001)
            RM3.train_RM()
            RM3.predict(scoring=scoring, reduce_by=reduce_by)
    
        try:
                
            print('-----')
            print
            RM4 = manage_RM(X_train=X2, y_train=y2, split_ratio=split_ratio, verbose=verbose, 
                            N_y_bins=N_y_bins, y_vects=y_vects)
            RM4.init_RM(max_iter=200, tol=0.001)
            RM4.train_RM()
            RM4.predict(scoring=scoring, reduce_by=reduce_by)
            print(RM4.N_train, RM4.N_in, RM4.N_train_y, RM4.N_out)
        except:
            print("!!! TEST 4 not passed !!!")
        
    
    return RM1, RM2, RM3, RM4
#%%
def test1():
    n_samples=10000
    y = np.random.uniform(-10, 10, n_samples)
    X = y**2
    solver = 'adam'#'lbfgs'
    activation = 'tanh'
    noise = 0.1
    X_test = np.linspace(0,100,1000)
    hidden_layer_sizes=(50,)
    
    RM1 = manage_RM(X_train=X, y_train=y, verbose=True, noise=noise, scaling=True,
                    RM_type='Keras')
    RM1.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=hidden_layer_sizes)
    RM1.train_RM()
    RM1.set_test(X_test, scaleit=True)
    RM1.predict(scoring=False)
    return RM1
#%%
def test_x2():
    import matplotlib.pyplot as plt
    n_samples=10000
    y = np.random.uniform(-10, 10, n_samples)
    X = y**2
    solver = 'adam'#'lbfgs'
    activation = 'tanh'
    noise = 0.1
    X_test = np.linspace(0,100,1000)
    hidden_layer_sizes=(50,)
    
    RM1 = manage_RM(X_train=X, y_train=y, verbose=True, noise=noise, scaling=True)
    RM1.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=hidden_layer_sizes)
    RM1.train_RM()
    RM1.set_test(X_test, scaleit=True)
    RM1.predict(scoring=False)
    
    RM2 = manage_RM(X_train=X, y_train=y, verbose=True, N_y_bins=100, noise=noise, scaling=True)
    RM2.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=hidden_layer_sizes)
    RM2.train_RM()
    RM2.set_test(X_test, scaleit=True)
    RM2.predict(scoring=False, reduce_by='max')
    
    RM3 = manage_RM(X_train=X, y_train=y, verbose=True, y_vects=np.linspace(-10,10,100), 
                    noise=noise, scaling=True)
    RM3.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=hidden_layer_sizes)
    RM3.train_RM()
    RM3.set_test(X_test, scaleit=True)
    RM3.predict(scoring=False, reduce_by='max')

    plt.scatter(X_test, RM1.pred, edgecolor='', c='b')
    plt.scatter(X_test, RM2.pred, edgecolor='', c='r')
    plt.scatter(X_test, RM3.pred, alpha=0.05, c='g')
    return RM1, RM2,RM3
#%%     
def test_x2_K():
    import matplotlib.pyplot as plt
    n_samples=3000
    y = np.random.uniform(-10, 10, n_samples)
    X = y**2
    solver = 'adam'
    activation = 'tanh'
    noise = 0.1
    X_test = np.linspace(0,100,1000)
    hidden_layer_sizes = (50,)
    
    RM1 = manage_RM(RM_type = 'KerasDis', X_train=X, y_train=y, verbose=True, 
                    noise=noise, y_vects=np.linspace(-10,10,100), min_discret=0, scaling=True)
    RM1.init_RM(solver=solver, activation=activation,
                hidden_layer_sizes=hidden_layer_sizes, epochs=100)
    RM1.train_RM()
    RM1.set_test(X_test, scaleit=True)
    RM1.predict(scoring=False, reduce_by='max')
    pred_max = RM1.pred
    RM1.predict(scoring=False, reduce_by='mean')
    pred_mean = RM1.pred
    plt.scatter(X_test, pred_max, edgecolor='', c='y')
    plt.scatter(X_test, pred_mean, edgecolor='', c='c')
    
    return RM1
#%%
def test_x3():
    import matplotlib.pyplot as plt
    n_samples=30000
    y = np.random.uniform(-10, 10, n_samples)
    X = y**3
    solver = 'lbfgs'
    activation = 'tanh'
    noise = None
    
    RM1 = manage_RM(X_train=X, y_train=y, verbose=True, noise=noise)
    RM1.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=(60, 60, 60))
    RM1.train_RM()
    RM1.set_test(np.linspace(-1000,1000,1000))
    RM1.predict(scoring=False)
    
    RM2 = manage_RM(X_train=X, y_train=y, verbose=True, N_y_bins=100, noise=noise)
    RM2.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=(60, 60, 60))
    RM2.train_RM()
    RM2.set_test(np.linspace(-1000,1000,1000))
    RM2.predict(scoring=False, reduce_by='max')

    RM3 = manage_RM(X_train=X, y_train=y, verbose=True, y_vects=np.linspace(-10,10,100), noise=noise)
    RM3.init_RM(max_iter=200000, tol=0.0000001, solver=solver, activation=activation,
                hidden_layer_sizes=(60, 60, 60))
    RM3.train_RM()
    RM3.set_test(np.linspace(-1000,1000,1000))
    RM3.predict(scoring=False, reduce_by='max')

    plt.scatter(RM1.X_test, RM1.pred)
    plt.scatter(RM2.X_test, RM2.pred)
    plt.scatter(RM3.X_test, RM3.pred, alpha=0.05)
    return RM1, RM2,RM3    
