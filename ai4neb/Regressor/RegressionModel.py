# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:03:31 2018

@author: christophemorisset
"""

# coding: utf-8
import numpy as np
import time
import random
from glob import glob
import pandas as pd

# my test 1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR,NuSVR 
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, SGDRegressor, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.utils import parallel_backend
try:
    import joblib
except:
    try:
        from sklearn.externals import joblib
    except:
        pass

# Keras
keras_access= 'Not installed'
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import backend as K
    from tensorflow.keras import initializers, regularizers
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    TF_OK = True
    keras_access = 'tf.keras'
except:
    try:
        import tensorflow as tf
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Dropout
        from keras.wrappers.scikit_learn import KerasRegressor
        from keras import backend as K
        from keras import initializers, regularizers
        TF_OK = True
        keras_access = 'keras'
    except:
        try:
            import tensorflow as tf
            from tensorflow.python.keras.models import Sequential, load_model
            from tensorflow.python.keras.layers import Dense, Dropout
            from tensorflow.python.keras import backend as K
            from tensorflow.python.keras import initializers, regularizers
            from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
            TF_OK = True
            keras_access = 'tf.python.keras'
        except:
            TF_OK = False
            
try:
    import xgboost as xgb
    XGB_OK = True
except:
    XGB_OK = False
    
    
try:
    import lightgbm as lgb
    LGB_OK = True
except:
    LGB_OK = False

try:
    import catboost
    CB_OK = True
except:
    CB_OK = False    
            
RM_version = "0.18"
#%% Main class
class manage_RM(object):
    """
    Manage Regression Model from SciKit learn and Tensorflow via Keras.
    """
    TF_OK = TF_OK
    keras_access = keras_access
    def __init__(self, RM_type = 'SK_ANN',
                 X_train=None, y_train=None, 
                 X_test=None, y_test=None,
                 scaling=False, 
                 scaling_y=False, 
                 pca_N=0,
                 use_log=False,
                 split_ratio = None,
                 verbose=False,
                 RM_filename=None,
                 use_data_manager=None,
                 use_RobustScaler=False,
                 random_seed=None,
                 N_y_bins=None,
                 y_vects=None,
                 min_discret=0,
                 max_discret=1,
                 clear_session=False,
                 notry=False,
                 predict_functional=False,
                 compile_=False):
        """
        Object to manage Regression Model(s).
            RM_type: can be 'SK_ANN', 'SK_ANN_Dis', 'SK_SVM', 'SK_NuSVM', 
                    'SK_BR', 'SK_AB', 'K_ANN', 'K_ANN_Dis', 'KSK_ANN', 'XGB' for now.
            X_train and y_train: are training sets, input and output repectively.
            X_test and y_test: are used for predictions.
            scaling and use_log: may be applied to X_train and X_test.
            scaling_y: can also be used. log(y) is not implemented (user can do it on her/his side).
            split_ratio: if not None, X_train and X_test are obtained by splitting X_test.
            pca_N: PCA can also be applied to X, if pca_N is > 0. pca_N is the number of components.
            RM_filename: is used to load a previously saved state.
            random_seed is sued to initialize the ANN
            if one of N_y_bins or y_vects is not None: discretization of the output is done
                N_y_bins: number of bins for the discretization. May be a tupple or a list
                y_vects: vector(s) on which the discretization is done
            min_discret [0] and max_discret [1]: range values for the discretization vectors
                
        """

        self.verbose = verbose
        if clear_session:
            try:
                K.clear_session()
            except:
                if self.verbose:
                    print('Session not cleared')
        self.init_random(random_seed)
        self.RM_version = RM_version
        if self.verbose:
            print('Instantiation. V {}'.format(self.RM_version))
        self.RM_type = RM_type
        self.scaling = scaling
        self.scaling_y = scaling_y
        self.use_log = use_log
        self.use_RobustScaler = use_RobustScaler 
        self.scaler = None
        self.scaler_y = None
        self.pca_N = pca_N
        self.pca = None
        self.train_scaled = False
        self.test_scaled = False
        self.isfin = None
        self.N_y_bins = N_y_bins
        self.y_vects = y_vects
        self.min_discret=min_discret
        self.max_discret=max_discret
        if split_ratio is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._copy_None(X_train),
                                                                                    self._copy_None(y_train),
                                                                                    test_size=split_ratio,
                                                                                    random_state=self.random_seed)
            if self.verbose:
                print('train and test sets are obtained by splitting X_train and y_train.')
                print('input test sets are not used.')
        else:
            self.X_train = self._copy_None(X_train)
            self.y_train = self._copy_None(y_train)
            self.X_test = self._copy_None(X_test)
            self.y_test = self._copy_None(y_test)
        self._init_dims(train=True, test=True)
        if self.verbose:
            print('Training set size = {}, Test set size = {}'.format(self.N_train, self.N_test))
        self.discretized = False
        if self.N_y_bins is not None or self.y_vects is not None:
            self.discretize()
            self._init_dims(train=True, test=True)
        else:
            self.y_train_ori = self.y_train
            self.y_test_ori = self.y_test            
        if self.scaling:
            self.scale_sets(use_log=self.use_log)
        else:
            self.X_train_unscaled = self.X_train
            self.X_test_unscaled = self.X_test
            self.y_train_unscaled = self.y_train
        self.RMs = None
        self.trained = False
        self._multi_predic = True
        self.RM_filename = RM_filename
        self.predict_functional = predict_functional
        if self.RM_filename is not None:
            self.load_RM(filename=self.RM_filename, notry=notry, compile_=compile_)
        if use_data_manager is not None:
            self.load_data_from_manager(use_data_manager)
        if self.verbose:
            print('Training set size = {}, Test set size = {}'.format(self.N_train, self.N_test))
        
    def init_random(self, seed):
        self.random_seed = seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        
    def _init_dims(self, train=True, test=True):
        
        def get_shape(a):
            if a is None:
                return 0, 0
            else:
                return a.shape
        if train:
            self.N_train, self.N_in = get_shape(self.X_train) 
            self.N_train_y, self.N_out = get_shape(self.y_train)
        if test:    
            self.N_test, self.N_in_test = get_shape(self.X_test)
            self.N_test_y, self.N_out_test = get_shape(self.y_test)
        
    def _len_None(self, v):
        if v is None:
            return 0
        else:
            return len(v)
        
    def _copy_None(self, v, add_dim=True):
        if v is None:
            to_return = None
        else:
            to_return = v.copy()
            if np.ndim(v) == 1 and add_dim:
                to_return = np.expand_dims(to_return, axis=1)
        return to_return
    
    def init_RM(self, user_defined_RM=None, **kwargs):
        """
        Initialisation of the Regression Model.
        user_defined_RM: an optional dictionnary containing a (list of) model(s),
            train_params, and _multi_predic.
        Any parameter is passed to the Model.
        self.N_out RM can be needed if not ANN type. 
        They are stored in self.RMs list.
        """
        

        if user_defined_RM is not None:
            if type(user_defined_RM) is not dict:
                raise TypeError('user_defined_RM needs to be a dictionnary')
            if 'model' not in user_defined_RM:
                raise ValueError('user_defined_RM dictionnary does not contain model')
            else:
                user_defined_model = user_defined_RM['model']
            if type(user_defined_model) is list:
                self.RMs = user_defined_model
            else:
                self.RMs = [user_defined_model]
            if 'train_params' in user_defined_RM:
                self.train_params = user_defined_RM['train_params']
            if '_multi_predic' in user_defined_RM:
                self._multi_predic = user_defined_RM['_multi_predic']
            return

        self.RMs = []
        self.train_params = {}
        self._multi_predic = False
        if self.RM_type in ('SK_ANN', 'SK_ANN_Dis'):
            self.RMs = [MLPRegressor(random_state=self.random_seed, **kwargs)]
            self._multi_predic = True
        elif self.RM_type == 'Poly':
            self.poly = []
            if "degree" in kwargs:
                degree = kwargs['degree']
            else:
                degree = 2
            for i in range(self.N_out):
                self.poly.append(PolynomialFeatures(degree=degree))
                self.RMs.append(LinearRegression())
        elif self.RM_type == 'SK_SVM':
            for i in range(self.N_out):
                self.RMs.append(SVR(**kwargs))
        elif self.RM_type == 'SK_NuSVM':
            for i in range(self.N_out):
                self.RMs.append(NuSVR(**kwargs))
        elif self.RM_type == 'SK_KRR':
            for i in range(self.N_out):
                self.RMs.append(KernelRidge(**kwargs))
        elif self.RM_type == 'SK_BR':
            for i in range(self.N_out):
                self.RMs.append(BayesianRidge(**kwargs))
        elif self.RM_type == 'SK_AB':
            for i in range(self.N_out):
                self.RMs.append(AdaBoostRegressor(random_state=self.random_seed,**kwargs))
        elif self.RM_type == 'SK_GBR':
            for i in range(self.N_out):
                self.RMs.append(GradientBoostingRegressor(random_state=self.random_seed,**kwargs))
        elif self.RM_type == 'SK_GPR':
            for i in range(self.N_out):
                self.RMs.append(GaussianProcessRegressor(random_state=self.random_seed,**kwargs))
            self._multi_predic = True
        elif self.RM_type == 'SK_SGDR':
            for i in range(self.N_out):
                self.RMs.append(SGDRegressor(random_state=self.random_seed,**kwargs))
        elif self.RM_type == 'SK_RFR':
            for i in range(self.N_out):
                self.RMs.append(RandomForestRegressor(random_state=self.random_seed,**kwargs))
            self._multi_predic = True
        elif self.RM_type in ("K_ANN", "K_ANN_Dis"):
            if not TF_OK:
                raise ValueError('Tensorflow not installed, Keras RM_type not available')
            def get_kwargs(kw, default):
                if kw in kwargs:
                    return kwargs[kw]
                else:
                    return default
            activation = get_kwargs('activation', 'relu')
            kernel_initializer = get_kwargs('kernel_initializer', 
                                            initializers.glorot_uniform(seed=self.random_seed))                
            bias_initializer = get_kwargs('bias_initializer', 
                                            initializers.glorot_uniform(seed=self.random_seed))
            optimizer = get_kwargs('optimizer', get_kwargs('solver', 'adam'))
            epochs = get_kwargs('epochs', 100)
            batch_size = get_kwargs('batch_size', None)
            validation_split = get_kwargs('validation_split', 0.0)
            hidden_layer_sizes = get_kwargs('hidden_layer_sizes', (10,10))
            random_state = get_kwargs('random_state', self.random_seed)
            dropout = get_kwargs('dropout', None)
            L1 = get_kwargs('L1', 0.)
            L2 = get_kwargs('L2', 0.)
            tf.compat.v1.random.set_random_seed(random_state)
            model = Sequential()
            model.add(Dense(hidden_layer_sizes[0], 
                            input_dim=self.N_in, 
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            activation=activation,
                            kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)))
            if dropout is None:
                d1 = 0.0
            else:
                if type(dropout) in (type(()), type([])):
                    d1 = dropout[0]
                else:
                    d1 = dropout
            if d1 != 0.0:
                model.add(Dropout(d1, seed=random_state, input_shape=(hidden_layer_sizes[0],)))
            for i_hl, hidden_layer_size in enumerate(hidden_layer_sizes[1:]):
                model.add(Dense(hidden_layer_size, 
                                activation=activation, 
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)))
                if dropout is not None:
                    if type(dropout) in (type(()), type([])):
                        di = dropout[i_hl+1]
                    else:
                        di = dropout
                    if di != 0.0:
                        model.add(Dropout(di, seed=random_state))
            if self.RM_type == 'K_ANN':
                model.add(Dense(self.N_out, 
                                activation='linear', 
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)))
                metrics = get_kwargs('metrics', ['mse','mae'])
                model.compile(loss='mse', 
                              optimizer=optimizer, 
                              metrics=metrics)
            elif self.RM_type == 'K_ANN_Dis':
                model.add(Dense(self.N_out, 
                                activation='softmax', 
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)))
                metrics = get_kwargs('metrics', ['accuracy'])
                model.compile(loss='categorical_crossentropy', 
                              optimizer=optimizer, 
                              metrics=metrics)
            if self.verbose:
                model.summary()
            self.RMs = [model]
            self.train_params = {'epochs': epochs, 
                                 'batch_size': batch_size, 
                                 'verbose': False, 
                                 'validation_split': validation_split}
            self._multi_predic = True
        elif self.RM_type == 'KSK_ANN':
            def get_kwargs(kw, default):
                if kw in kwargs:
                    return kwargs[kw]
                else:
                    return default
            activation = get_kwargs('activation', 'relu')
            kernel_initializer = get_kwargs('kernel_initializer', 
                                            initializers.glorot_uniform(seed=self.random_seed))
            bias_initializer = get_kwargs('bias_initializer', 
                                            initializers.glorot_uniform(seed=self.random_seed))
            optimizer = get_kwargs('optimizer', get_kwargs('solver', 'adam'))
            epochs = get_kwargs('epochs', 1)
            batch_size = get_kwargs('batch_size', None)
            validation_split = get_kwargs('validation_split', 0.0)
            hidden_layer_sizes = get_kwargs('hidden_layer_sizes', (10,10))
            random_state = get_kwargs('random_state', self.random_seed)
            tf.random.set_random_seed(random_state)
            
            def create_model(hidden_layer_sizes, N_in, activation, random_state,
                             N_out):
                model = Sequential()
                model.add(Dense(hidden_layer_sizes[0], 
                                input_dim=N_in, 
                                activation=activation))
                for hidden_layer_size in hidden_layer_sizes[1:]:
                    model.add(Dense(hidden_layer_size, 
                                    activation=activation,))
                model.add(Dense(N_out, 
                                activation='linear'))
                metrics = ['mse','mae']
                model.compile(loss='mse', 
                              optimizer=optimizer, 
                              metrics=metrics)
                return model
                
            model = KerasRegressor(create_model, 
                                   hidden_layer_sizes = hidden_layer_sizes, 
                                   N_in = self.N_in,
                                   activation=activation,  
                                   random_state=random_state,
                                   N_out=self.N_out)
            #if self.verbose:
            #    model.summary()
            self.RMs = [model]
            self.train_params = {'epochs': epochs, 
                                 'batch_size': batch_size, 
                                 'verbose': False, 
                                 'validation_split': validation_split}
            self._multi_predic = True # TBC ***
        elif self.RM_type == 'XGB':
            if not XGB_OK:
                raise ValueError('xgboost not installed')
            for i in range(self.N_out):
                self.RMs.append(xgb.XGBRegressor(random_state=self.random_seed, **kwargs))
        elif self.RM_type == 'LGB':
            if not LGB_OK:
                raise ValueError('lightgbm not installed')
            for i in range(self.N_out):
                self.RMs.append(lgb.LGBMRegressor(random_state=self.random_seed, **kwargs))
        elif self.RM_type == 'CatBoost':
            if not CB_OK:
                raise ValueError('CatBoost not installed')
            for i in range(self.N_out):
                self.RMs.append(catboost.CatBoostRegressor(random_state=self.random_seed, **kwargs))
        else:
            raise ValueError('Unkown Regression method {}'.format(self.RM_type))
        if self.verbose:
            print('Regression Model {}'.format(self.RM_type))

    def __add_noise(self, only_test=False):
        """
        Obsolete, noise has to be managed outside the class.
        """
        if self.noise is not None:
            if type(self.noise) in (tuple, list):
                noise_train = self.noise[0]
                noise_test = self.noise[1]
            else:
                noise_train = self.noise
                noise_test = self.noise
                
            if not only_test:
                self.X_train *= 1 + np.random.normal(0.0, noise_train, self.X_train.shape)
                train_str = ' and {} to train'.format(noise_train)
            else:
                train_str = ''
            
            if self.X_test is not None:
                self.X_test *= 1 + np.random.normal(0.0, noise_test, self.X_test.shape)
                test_str = ' {} to test'.format(noise_test)
            else:
                test_str = ''
            
            if self.verbose:
                print("Adding noise{}{}.".format(test_str, train_str))
            
    def discretize(self):
        
        if self.discretized:
            raise ValueError('Can not discretize twice')
            
        self.y_train_ori = self._copy_None(self.y_train)
        self.y_test_ori = self._copy_None(self.y_test)

        if self.y_vects is None:
            if type(self.N_y_bins) is int:
                self.N_y_bins = np.array([self.N_y_bins] * self.N_out)
            
            self.minmax = np.percentile(self.y_train, (2.5, 97.5), axis=0)
            self.deltas = self.minmax[1,:] - self.minmax[0,:]
            if self.N_out == 1:
                self.y_vects = np.linspace(self.minmax[0], self.minmax[1], self.N_y_bins[0])
            else:
                self.y_vects = []
                for i in np.arange(self.N_out):
                    self.y_vects.append(np.linspace(self.minmax[0,i], self.minmax[1,i], self.N_y_bins[i]))
                self.y_vects = np.array(self.y_vects)
        else:
            #if np.ndim(self.y_vects) == 1:
            #    self.y_vects = np.expand_dims(self.y_vects, axis=1)
            if type(self.y_vects) not in (list, tuple):
                self.N_y_bins = np.array([len(self.y_vects)] * self.N_out)
                self.y_vects = np.array([self.y_vects] * self.N_out)
                if self.N_out == 1:
                    self.y_vects = self.y_vects.T
            else:
                self.N_y_bins = np.array([len(y_vect) for y_vect in self.y_vects])
                self.y_vects = np.array(self.y_vects)
            self.minmax = np.array((self.y_vects.min(0), self.y_vects.max(0)))
            self.deltas = self.minmax[1,:] - self.minmax[0,:]
        self.y_train = self._discretize1(self.y_train)
        self.y_test = self._discretize1(self.y_test)
        self.discretized = True
        
    def _discretize1(self, y):
        
        if y is None:
            return None
        new_y = self.min_discret * np.ones((y.shape[0], self.N_y_bins.sum()))
        
#        for i in np.arange(self.N_out):
        for i in np.arange(y.shape[1]):
            tt = np.round((y[:,i] - self.minmax[0,i]) / self.deltas[i] * (self.N_y_bins[i] - 1)).astype(int)
            tt = np.where(tt < 0, 0, tt)
            tt = np.where(tt > self.N_y_bins[i]-1, self.N_y_bins[i]-1, tt)
            if i > 0:
                tt += np.cumsum(self.N_y_bins)[i-1]
            new_y[(np.arange(y.shape[0]), tt)] = self.max_discret
#            if self.verbose:
#                print("Discretizing column {} on {} bins".format(i, self.N_y_bins[i])) 
        return new_y
            
    def __set_train(self, X=None, y=None):
        """
        Obsolete. It is actually not useful!
        """
        self.X_train = self._copy_None(X)
        self.y_train = self._copy_None(y)
        self.train_scaled = False
        self.discretized = False
        if self.N_y_bins is not None or self.y_vects is not None:
            self.discretize()
            self._init_dims(train=True, test=True)
        if self.scaling:
            self.scale_sets(use_log=self.use_log)
        else:
            self.scaler = None
        self._init_dims(train=True, test=False)
        self.RMs = None

    def set_test(self, X=None, y=None):

        self.X_test = self._copy_None(X)
        self.y_test = self._copy_None(y)
                
        self.test_scaled = False
        if self.scaling:
            self.scale_sets(use_log=self.use_log)
        else:
            self.X_test_unscaled = self.X_test
        self.discretized = False
        if self.N_y_bins is not None or self.y_vects is not None:
            self.y_test_ori = self._copy_None(self.y_test)                
            self.y_test = self._discretize1(self.y_test)
        else:
            self.y_test_ori = self.y_test            
        self._init_dims(train=False, test=True)

    def _log_data(self, X, y):
        """
        Apply log10 to X. 
        Filter X and y to isfinite(X)
        Return filtered X and y
        """
        if X is None:
            return None, None
        else:
            n_keys = X.shape[1]
            with np.errstate(invalid='ignore', divide='ignore'):
                X = np.log10(X)
            self.isfin = np.isfinite(X).sum(1) == n_keys
            X = X[self.isfin]
            if y is not None:
                y = y[self.isfin]
            
            return X, y
        
    def _set_scaler(self, force=False):
        if (self.scaler is None) or force:
            if self.use_RobustScaler:
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(self.X_train)
        
        if self.pca_N != 0:
            self.pca = PCA(n_components=self.pca_N)
            self.pca.fit(self.scaler.transform(self.X_train))

    def _set_scaler_y(self, force=False):
        if ((self.scaler_y is None) or force) and self.scaling_y and self.y_train is not None:
            if self.use_RobustScaler:
                self.scaler_y = RobustScaler()
            else:
                self.scaler_y = StandardScaler()
            self.scaler_y.fit(self.y_train)

    def scale_sets(self, force=False, use_log=False):
        """
        A scaler is created in self.scaler if it does not already exist.
        (it may have been recovered from saved file)
        It is fit to self.X_train if it is just created. Otherwise is 
        is supposed to already have been fit. To restart fitting, set self.scaler to None.
        If use_log, the self.X_train and self.X_test data are transformed to log10
        and only finite data is kept.
        The scaler is applied to self.X_train and self.X_test if they exist.
        """
        self.y_train_unscaled = self.y_train
        log_str = ''
        pca_str = ''
        if (not self.train_scaled) or force:
            self.X_train_unscaled = self._copy_None(self.X_train)
            if use_log:
                self.X_train, self.y_train = self._log_data(self.X_train, self.y_train)
                log_str = 'Log10 applied. '
            if self.X_train is not None:
                self._set_scaler()
                self._set_scaler_y()
                self.X_train = self.scaler.transform(self.X_train)
                if self.pca is not None:
                    self.X_train = self.pca.transform(self.X_train)
                    pca_str = 'PCA {} components applied. '.format(self.pca_N)
                if self.scaling_y and self.scaler_y is not None and self.y_train is not None:
                    self.y_train_unscaled = self._copy_None(self.y_train)     
                    self.y_train = self.scaler_y.transform(self.y_train)
                self._init_dims(train=True, test=False)
                self.train_scaled = True
            
            if self.verbose:
                print('Train data scaled. {}{}'.format(log_str,pca_str))
        
        self.y_test_unscaled = self.y_test
        log_str = ''
        pca_str = ''
        if (not self.test_scaled) or force:
            if self.X_test is not None:
                self.X_test_unscaled = self._copy_None(self.X_test)
                if use_log:
                    self.X_test, self.y_test = self._log_data(self.X_test, self.y_test)
                    log_str = 'Log10 applied. '
                if self.X_test is not None:
                    self.X_test = self.scaler.transform(self.X_test)
                    if self.pca is not None:
                        self.X_test = self.pca.transform(self.X_test)
                        pca_str = 'PCA {} components applied. '.format(self.pca_N)
                    # No need to scale y_test as pred is inverse scaled after determined.
                    #if self.scaling_y and self.scaler_y is not None and self.y_test is not None:
                    #    self.y_test = self.scaler_y.transform(self.y_test)
                    self._init_dims(train=False, test=True)
                    self.test_scaled = True
            
            if self.verbose:
                print('Test data scaled. {}{}'.format(log_str,pca_str))
                
        if self.verbose:
            print('Training set size = {}, Test set size = {}'.format(self.N_train, self.N_test))

    def train_RM(self, p_backend='loky', scoring=True):
        """
        Training the models.
        """
        start = time.time()
        if not self.train_scaled and self.verbose:
            if self.verbose:
                print('WARNING: training data not scaled')
        self.train_score = []
        self.history = []
        if self.N_train != self.N_train_y:
            raise Exception('N_train {} != N_train_y {}'.format(self.N_train,
                            self.N_train_y))
        if self.verbose:
            print('Training {} inputs for {} outputs with {} data'.format(self.N_in,
                  self.N_out, self.N_train_y))
        if self._multi_predic:
            RM = self.RMs[0]
            if self.y_train.ndim == 1:
                y_train = np.ravel(self.y_train)
            elif self.y_train.ndim == 2 and self.y_train.shape[1] == 1:
                y_train = np.ravel(self.y_train)
            else:
                y_train = self.y_train
            history = RM.fit(self.X_train, y_train, **self.train_params)
            self.history = [history]
            if scoring:
                train_score = score(RM, self.X_train, y_train)
                self.train_score = [train_score]
                iter_str = '.'
                if self.verbose:
                    try:
                        iter_str = ', with {} iterations.'.format(RM.n_iter_)
                    except:
                        iter_str = '.'
                    print('RM trained{} Score = {:.3f}'.format(iter_str, train_score))
        else:
            if self.y_train.ndim == 1:
                y_trains = (self.y_train,)
            elif self.y_train.ndim == 2 and self.y_train.shape[1] == 1:
#                y_trains = np.ravel(self.y_train)
                y_trains = self.y_train.T
            else:
                y_trains = self.y_train.T
            i_RM = 0
            for RM, y_train in zip(self.RMs, y_trains):
                if self.RM_type == 'Poly':
                    to_fit = self.poly[i_RM].fit_transform(self.X_train)
                else:
                    to_fit = self.X_train
                i_RM += 1
                history = RM.fit(to_fit, y_train, **self.train_params)
                self.history.append(history)
                if scoring:
                    train_score = score(RM, to_fit, y_train)
                    self.train_score.append(train_score)
                    iter_str = '.'
                    if self.verbose:
                        try:
                            iter_str = ', with {} iterations.'.format(RM.n_iter_)
                        except:
                            iter_str = '.'
                        print('RM trained{} Score = {:.3f}'.format(iter_str, train_score))

        self.trained = True
        end = time.time()
        self.training_time = end - start
        if self.verbose:
            for RM in self.RMs:
                print(RM)
            print('Training time {:.1f} s.'.format(self.training_time))

    def _norm_pred(self):
        self.pred_ori = self._copy_None(self.pred)
        tmp = self.pred - np.expand_dims(self.pred.min(1), axis=1)
        self.pred_norm =  tmp / np.expand_dims(tmp.sum(1), axis=1)
                
    def plot_loss(self, ax=None, i_RM=0, **kwargs):
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            f, ax = plt.subplots()
        if self.RMs is not None:
            for RM in self.RMs:
                if self.RM_type[0:3] == 'SK_':
                    self.loss_values = RM.loss_curve_
                    val_loss_values = None
                elif self.RM_type[0:2] == 'K_':
                    self.loss_values = self.history[i_RM].history['loss']
                    try:
                        val_loss_values = self.history[i_RM].history['val_loss']
                    except:
                        val_loss_values = None   
                ax.plot(self.loss_values, label='Train loss', **kwargs)
                if val_loss_values is not None:
                    ax.plot(val_loss_values, label='Validation loss')
            ax.set_yscale('log')
    
    def predict(self, scoring=False, reduce_by=None, y_str=None, X_str=None, **kwargs):
        """
        Compute the prediction using self.X_test
        Results are stored into self.pred
        if scoring, a score is computed comparing with self.y_test
        """
        start = time.time()
        if self.RMs is None:
            raise Exception('WARNING: Regression Model not set up')
        if not self.trained:
            raise Exception('WARNING: Regression Model not trained')
        if not self.test_scaled and self.verbose:
            print('WARNING: test data not scaled')
        if self._multi_predic:
            if self.predict_functional:
                self.pred = self.RMs[0](self.X_test)[0,::]                
            else:
                self.pred = self.RMs[0].predict(self.X_test, **kwargs)
        else:
            self.pred = []
            for i_RM, RM in enumerate(self.RMs):
                if self.RM_type == 'Poly':
                    to_predict = self.poly[i_RM].fit_transform(self.X_test)
                else:
                    to_predict = self.X_test
                if self.predict_functional:
                    self.pred.append(RM(to_predict))[0,::]
                else:
                    self.pred.append(RM.predict(to_predict, **kwargs))
            self.pred = np.array(self.pred).T
        if scoring:
            if self.N_test != self.N_test_y:
                raise Exception('N_test {} != N_test_y {}'.format(self.N_test, self.N_test_y))
            if self._multi_predic:
                try:
                    self.predic_score = score(self.RMs[0], to_predict, self.y_test, axis=0, 
                                              predict_functional=self.predict_functional)  #[score(RM, self.X_test, self.y_test) for RM in self.RMs]
                except:
                    self.predic_score = [np.nan for RM in self.RMs]
            else:
                if self.y_train.ndim == 1:
                    y_tests = (self.y_test,)
                else:
                    y_tests = self.y_test.T
                try:
                    self.predic_score = [score(RM, self.X_test, y_test) for RM, y_test in zip(self.RMs, y_tests)]
                except:
                    self.predic_score = [np.nan for RM, y_test in zip(self.RMs, y_tests)]
            if self.N_out != self.N_out_test:
                raise Exception('N_out {} != N_out_test {}'.format(self.N_out,
                                self.N_out_test))
            if self.verbose:
                print('Score = {}'.format(', '.join(['{:.3f}'.format(ts) for ts in self.predic_score])))
        if self.scaling_y:
            self.pred = self.scaler_y.inverse_transform(self.pred)
        self.pred = np.asarray(self.pred) # in case of a Tensor
        if reduce_by is not None:
            if self.N_y_bins is None:
                raise Exception('Can not reduce if N_y_bins is not defined.')
            self._norm_pred()
            
            if len(self.N_y_bins) == 1:
                self.pred_mean = np.dot(self.pred,self.y_vects)
                self.pred_mean_norm = np.dot(self.pred_norm,self.y_vects)
                self.pred_max = self.y_vects[np.argmax(self.pred_norm, 1)]
            else:
                self.pred_mean = np.zeros((self.N_test, len(self.N_y_bins)))
                self.pred_mean_norm = np.zeros((self.N_test, len(self.N_y_bins)))
                self.pred_max = np.zeros((self.N_test, len(self.N_y_bins)))
                for i in np.arange(len(self.N_y_bins)):
                    if i == 0:
                        i_inf = 0
                    else:
                        i_inf = self.N_y_bins.cumsum()[i-1]
                    i_sup = self.N_y_bins.cumsum()[i]
                    self.pred_mean[:,i] = np.dot(self.pred[:,i_inf:i_sup],self.y_vects[i])
                    self.pred_mean_norm[:,i] = np.dot(self.pred_norm[:,i_inf:i_sup],self.y_vects[i])
                    self.pred_max[:,i] = self.y_vects[i, np.argmax(self.pred_norm[:,i_inf:i_sup], 1)]
            
            if reduce_by == 'mean':
                self.pred = self.pred_mean
                print('Reducing y by mean')
            elif reduce_by == 'mean_norm':
                self.pred = self.pred_mean_norm
                print('Reducing y by mean')
            elif reduce_by == 'max':
                self.pred = self.pred_max
                print('Reducing y by max')
        end = time.time()
        if y_str is not None and X_str is not None:
            #print(type(self.X_test_unscaled), type(self.pred.shape))
            self.pred_df = pd.concat([pd.DataFrame(np.asarray(self.X_test_unscaled), columns = X_str),
                           pd.DataFrame(self.pred, columns = y_str)], axis = 1)
        if self.verbose:
            print('Predicting from {} inputs to {} outputs using {} data in {:.2f} secs.'.format(self.N_in_test,
                  self.N_out, self.N_test, end - start))
        
    def save_RM(self, filename='RM',
                save_train=False, save_test=False,
                data_filename = None,
                X_str = None, y_str = None,
                **kwargs):
        """
        Save the following values:
            self.RM_version, self.RM_type, 
                   self.scaling, self.scaling_y, 
                   self.use_log, self.use_RobustScaler,
                   self.train_scaled, self.test_scaled,
                   self.scaler, self.scaler_y, 
                   self.pca_N, self.pca, 
                   self.N_y_bins, self.y_vects,
                   self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                   self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                   self.train_score, self._multi_predic,
                   self.trained, self.training_time,
                   self.random_seed,
                   self.RMs
        
        If using Keras, self.RMs is saved as None, as the RMs are stored in a different file.
        
        If either save_train or save_test is True, the corresponding dataset will be saved as a pandas
        DataFrame with the name [data_filename]_data.gzip.
        """
                
        if not self.trained:
            raise Exception('Regression Model not trained')
        if save_train:
            X_train, y_train = self.X_train_unscaled, self.y_train_unscaled
        else:
            X_train, y_train = None, None
        if save_test:
            X_test, y_test = self.X_test_unscaled, self.y_test_unscaled
        else:
            X_test, y_test = None, None
        
        to_save = [self.RM_version, self.RM_type,
                   self.scaling, self.scaling_y, 
                   self.use_log, self.use_RobustScaler,
                   self.train_scaled, self.test_scaled,
                   self.scaler, self.scaler_y, 
                   self.pca_N, self.pca, 
                   self.N_y_bins, self.y_vects,
                   self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                   self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                   self.train_score, self._multi_predic,
                   self.trained, self.training_time,
                   self.random_seed 
                   ]
        
        if save_train or save_test:
            if data_filename is None:
                data_filename = filename
            data_manager = manage_RM_data(verbose = self.verbose)
            data_manager.save_data(data_filename + "_data.gzip", X_train, y_train, X_test, y_test, X_str, y_str)
        
        if self.RM_type[0:3] == 'SK_': 
            to_save.append(self.RMs)
            joblib.dump(to_save, filename+'.ai4neb_sk', **kwargs)
            if self.verbose:
                print('RM save to {}.ai4neb_sk'.format(filename))
        elif self.RM_type[0:2] == 'K_':
            to_save.append(None)
            joblib.dump(to_save, filename+'.ai4neb_k0')
            if self.verbose:
                print('RM save to {}.ai4neb_k0'.format(filename))
            for i, RM in enumerate(self.RMs):
                RM.save('{}.ai4neb_k{}'.format(filename, i+1), **kwargs)
                if self.verbose:
                    print('RM save to {}.ai4neb_k{}'.format(filename, i+1))
        elif self.RM_type == 'XGB':
            to_save.append(None)
            joblib.dump(to_save, filename+'.ai4neb_xgb0')
            if self.verbose:
                print('RM save to {}.ai4neb_xgb0'.format(filename))
            for i, RM in enumerate(self.RMs):
                RM.save_model('{}.ai4neb_xgb{}'.format(filename, i+1), **kwargs)
                if self.verbose:
                    print('RM save to {}.ai4neb_xgb{}'.format(filename, i+1))
        elif self.RM_type == 'CatBoost':
            to_save.append(None)
            joblib.dump(to_save, filename+'.ai4neb_catb0')
            if self.verbose:
                print('RM save to {}.ai4neb_catb0'.format(filename))
            for i, RM in enumerate(self.RMs):
                RM.save_model('{}.ai4neb_catb{}'.format(filename, i+1), **kwargs)
                if self.verbose:
                    print('RM save to {}.ai4neb_catb{}'.format(filename, i+1))
        else:
           print('Do not know how to save {} machine'.format(self.RM_type))
            
    def load_RM(self, filename='RM', notry=False, compile_=False):
        """
        Loading previously saved model.
        joblib is used to load.
        A WARNING is issued if a different version is found in the file.
        
        Example: Only X_test is needed.
        
        RM = manage_RM(X_test=X_test)
        RM.load_RM(filename)
        
        it can also be included to the instantiation:
            
        RM = manage_RM(X_test=X_test, RM_filename=RM_filename)
        RM.scale_sets(use_log=True)
        RM.predict(scoring=False)
        
        """
        files = glob("{}.*".format(filename))
        
        format_to_read = None
        if "{}.ai4neb_sk".format(filename) in files:
            to_read = "{}.ai4neb_sk".format(filename)
            format_to_read = 'SK'
        elif "{}.ai4neb_k0".format(filename) in files: 
            to_read = "{}.ai4neb_k0".format(filename)
            format_to_read = 'K'
        elif "{}.ai4neb_xgb0".format(filename) in files: 
            to_read = "{}.ai4neb_xgb0".format(filename)
            format_to_read = 'XGB'
        elif "{}.ai4neb_catb0".format(filename) in files: 
            to_read = "{}.ai4neb_catb0".format(filename)
            format_to_read = 'CatBoost'
        else:
            to_read = None
            print('No ai4neb file found for {}'.format(filename))
            self.model_read = False
            return
        
        if notry:
            RM_tuple = joblib.load(to_read)
            if self.verbose:
                print('RM loaded from {}'.format(to_read))
        else:
            try:
                RM_tuple = joblib.load(to_read)
                if self.verbose:
                    print('RM loaded from {}'.format(to_read))
            except:
                print('!! ERROR reading {}'.format(to_read))
        
        load_version = RM_tuple[0]
        if self.RM_version != load_version and self.verbose:
            print('WARNING: version loaded from {} is {}. Version from RM class is {}.'.format(to_read, 
                                                                  load_version, self.RM_version))
        
        if load_version in ("0.17"):
            (self.RM_version, self.RM_type, 
                   X_train, y_train, X_test, y_test,
                   self.X_train, self.y_train, self.X_test, self.y_test,
                   self.scaling, self.scaling_y, 
                   self.use_log, self.use_RobustScaler,
                   self.train_scaled, self.test_scaled,
                   self.scaler, self.scaler_y, 
                   self.pca_N, self.pca, 
                   self.N_y_bins, self.y_vects,
                   self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                   self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                   self.train_score, self._multi_predic,
                   self.trained, self.training_time,
                   self.random_seed, 
                   self.RMs) = RM_tuple
            
        elif load_version in ("0.18"):
            (self.RM_version, self.RM_type, 
                   self.scaling, self.scaling_y, 
                   self.use_log, self.use_RobustScaler,
                   self.train_scaled, self.test_scaled,
                   self.scaler, self.scaler_y, 
                   self.pca_N, self.pca, 
                   self.N_y_bins, self.y_vects,
                   self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                   self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                   self.train_score, self._multi_predic,
                   self.trained, self.training_time,
                   self.random_seed, 
                   self.RMs) = RM_tuple
            
        else:
            self.model_read = False
            print('!! ERROR. This version is not supported.')
        if format_to_read == 'K':
            if notry:
                self.RMs = [load_model("{}.ai4neb_k1".format(filename))]
                if self.verbose:
                    print('RM loaded from {}.ai4neb_k1'.format(filename))
            else:
                try:
                    self.RMs = [load_model("{}.ai4neb_k1".format(filename))]
                    if self.verbose:
                        print('RM loaded from {}.ai4neb_k1'.format(filename))
                except:
                    self.model_read = False
                    print('!! ERROR reading {}.ai4neb_k1'.format(filename))
        elif format_to_read == 'XGB':
            try:
                self.RMs = []
                for i in np.arange(self.N_out)+1:
                    self.RMs.append(xgb.XGBRegressor())
                    self.RMs[i-1].load_model('{}.ai4neb_xgb{}'.format(filename, i))
                    if self.verbose:
                        print('RM loaded from {}.ai4neb_xgb{}'.format(filename, i))
            except:
                self.model_read = False
                print('!! ERROR reading {}.ai4neb_xgb1'.format(filename))
                
        elif format_to_read == 'CatBoost':
            try:
                self.RMs = []
                for i in np.arange(self.N_out)+1:
                    self.RMs.append(xgb.XGBRegressor())
                    self.RMs[i-1].load_model('{}.ai4neb_catb{}'.format(filename, i))
                    if self.verbose:
                        print('RM loaded from {}.ai4neb_catb{}'.format(filename, i))
            except:
                self.model_read = False
                print('!! ERROR reading {}.ai4neb_catb1'.format(filename))
                
        self.discretized = False

        if compile_:
            if self.verbose:
                print('Compiling model')
            for rm in self.RMs:
                rm.compile()        
        if self.N_y_bins is not None or self.y_vects is not None:
            self.discretize()
        else:
            self.y_train_ori = self.y_train
            self.y_test_ori = self.y_test            
        if self.scaling:
            self.scale_sets(use_log=self.use_log, force=True)
        else:
            self.X_train_unscaled = self.X_train
            self.X_test_unscaled = self.X_test
            self.y_train_unscaled = self.y_train
        self.model_read = True
        
    def load_data_from_manager(self, manager):
        self.X_train, self.y_train, self.X_test, self.y_test = manager.X_train, manager.y_train, manager.X_test, manager.y_test
        
        if self.scaling:
            self.scale_sets(use_log=self.use_log, force=True)
        else:
            self.X_train_unscaled = self.X_train
            self.X_test_unscaled = self.X_test
            self.y_train_unscaled = self.y_train

        
class manage_data():
    def __init__(self, filename = None, dataframe = None,
                 X= None, y= None,
                 X_str = None, y_str = None, 
                 split_ratio = 0.2, 
                 verbose = False):
        """
        The user send the data in one of the (exclusive) following ways:
        - filename: a csv file containing X and y.
        - dataframe: pandas o structured numpy table, containing X and y.
        - X, and y: pandas o [structured] numpy tables.  
        """
        self.X_str = X_str
        self.y_str = y_str
        self.split_ratio = split_ratio
        self.verbose = verbose
        
        if filename is not None:
            self.load_data(filename)
        elif dataframe is not None:
            if type(dataframe) is not pd.core.frame.DataFrame:
                self.data = pd.DataFrame(dataframe)
            else:
                self.data = dataframe
            if X_str is None or y_str is None:
                raise Exception(f'X_str and y_str must be defined')
            if len(X_str) + len(y_str) != len(self.data.columns):
                raise Exception(f'X_str and y_str must sum the number of dataframe columns')
            if self.X_str is not None:
                for k in self.X_str:
                    if k not in self.data.keys():
                        raise Exception(f'{k} is not a column name of dataframe')

        elif X is not None and y is not None:
            if len(X) != len(y):
                raise Exception('X and y are not the same length')
            X = self.transfo_X('X', X)
            y = self.transfo_X('y', y)
            self.data = pd.concat([X, y], axis=1)
        else :
            raise Exception('You need to provide some data...')

    def transfo_X(self, type_, data):

        if type_ == 'X':
            type_str = self.X_str
        elif type_ == 'y':
            type_str = self.y_str

        if type(data) is pd.core.frame.DataFrame:
            if type_str is not None:
                for k in type_str:
                    if k not in data.keys():
                        raise Exception(f'{k} is not a column name of input')
            else:
                data.columns = [f"{type_}_{i}" for i in range(len(data.columns))]
                print('Column names have been updated')
                type_str = data.columns.values
        elif type(data) is np.ndarray:
            if data.dtype.names is None:
                data.columns = [f"{type_}_{i}" for i in range(len(data.dtype))]
                print('Column names have been updated')
                type_str = data.columns.values
                data = pd.DataFrame(data, columns = type_str)
            else:
                if self.data_str is not None:
                    for k in type_str:
                        if k not in data.dtype.names:
                            raise Exception(f'{k} is not a column name of input')
                else:
                    type_str = data.dtype.names
                data = pd.DataFrame(data)
        if type_ == 'X':
            self.X_str = type_str
        elif type_ == 'y':
            self.y_str = type_str
        
        return data

                


            
    def load_data(self, filename):

        # Check existence of the file
        self.data = pd.read_csv(filename)

        if self.X_str is None:
            print("X_str not defined by user. Searching default labels...")
            X_str = [label for label in self.data.columns if label[:2] == "X_"]
            if not X_str:
                raise Exception("Default labels not found for X in file {}. Please specify which columns to use in X_str.".format(filename))

            self.X_str = X_str
            print("Default labels found for X. Using X_str = {}".format(X_str))
        if self.y_str is None:
            print("y_str not defined by user. Searching default labels...")
            y_str = [label for label in self.data.columns if label[:2] == "y_"]
            if not y_str:
                raise Exception("Default labels not found for y in file {}. Please specify which columns to use in y_str.".format(filename))


            self.y_str = y_str
            print("Default labels found for y. Using y_str = {}".format(y_str))
        if self.verbose:
                print("Data loaded from {}".format(filename))

    def select_test(self):

        if 'train' not in self.data.keys():
            # Add a column "train" set to 0 or 1 for test or train resp.
            N = len(self.data)
            self.data['train'] = np.ones(N)
            self.data['train'].iloc[np.random.choice(N, int(N*self.split_ratio))] = 0

        if len(self.X_train) == 0:
            self.X_train, self.y_train = None, None

        if len(self.X_test) == 0:
            self.X_test, self.y_test = None, None

    @property
    def X_train(self):
        return self.data.loc[self.data['train'] == 1, self.X_str]
    @property
    def y_train(self):
        return self.data.loc[self.data['train'] == 1, self.y_str]
    @property
    def X_test(self):
        return self.data.loc[self.data['train'] != 1, self.X_str]
    @property
    def y_test(self):
        return self.data.loc[self.data['train'] != 1, self.y_str]



        
    def save_data(self, filename, X_train, y_train, X_test, y_test, X_str = None, y_str = None):
        if X_train is None and y_train is None:
            X_train, y_train = np.array([]), np.array([])
        if X_test is None and y_test is None:
            X_test, y_test = np.array([]), np.array([])
        
        if X_str is None:
            x_columns = self._get_shape(X_train)[1]
            if x_columns == 0:
                x_columns = self._get_shape(X_test)[1]
                
            X_str = ["X_" + str(i) for i in range(x_columns)]
            
        if y_str is None:
            y_columns = self._get_shape(y_train)[1]
            if y_columns == 0:
                y_columns = self._get_shape(y_test)[1]
                
            y_str = ["y_" + str(i) for i in range(y_columns)]
        
        train = pd.concat([pd.DataFrame(X_train, columns = X_str),
                           pd.DataFrame(y_train, columns = y_str)], axis = 1)
        train["train"] = np.ones(len(train))
        
        test = pd.concat([pd.DataFrame(X_test, columns = X_str),
                           pd.DataFrame(y_test, columns = y_str)], axis = 1)
        test["train"] = np.zeros(len(test))
        
        data = pd.concat([train, test], ignore_index=True)
        
        data.to_csv(filename, compression = "gzip", index = False)
        if self.verbose:
            print("Data saved to {}".format(filename))
    
    def _get_shape(self, X):
        if X is None:
            rows = 0
            columns = 0
        else:
            rows = X.shape[0]
            try:
                columns = X.shape[1]
            except IndexError:
                columns = 1
            
        return (rows, columns)

    
def score(RM, X, y_true, axis=None, predict_functional=False):
    """
    (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() 
    and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
    """
    if predict_functional:
        y_pred = RM(X)[0,::]
    else:
        y_pred = RM.predict(X)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)
    u = ((y_true - y_pred) ** 2).sum(axis=axis)
    v = ((y_true - y_true.mean()) ** 2).sum(axis=axis)
    
    return 1 - u/v
                
#%% __main__
if __name__ == "__main__":
    pass
    