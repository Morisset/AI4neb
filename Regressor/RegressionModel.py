# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:03:31 2018

@author: christophemorisset
"""

# coding: utf-8

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR,NuSVR 
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA

# Keras
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import backend as K
    TF_OK = True
except:
    try:
        import tensorflow as tf
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense, Dropout
        from tensorflow.python.keras import backend as K
        #from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
        TF_OK = True
    except:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras import backend as K
            TF_OK = True
        except:
            TF_OK = False
version = "0.15"
    
#%%
class manage_RM(object):
    """
    Manage Regression Model from SciKit learn and Tensorflow via Keras.
    """
    TF_OK = TF_OK
    def __init__(self, RM_type = 'ANN',
                 X_train=None, y_train=None, 
                 X_test=None, y_test=None,
                 scaling=False, 
                 scaling_y=False, 
                 pca_N=0,
                 use_log=False,
                 split_ratio = None,
                 verbose=False,
                 RM_filename=None, use_RobustScaler=False,
                 random_seed=None,
                 noise=None,
                 N_y_bins=None,
                 y_vects=None,
                 min_discret=0,
                 max_discret=1,
                 clear_session=False):
        """
        Object to manage Regression Model(s).
            RM_type: can be "ANN", "SVM", "NuSVM", "BR", "AB" for now.
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
        if clear_session:
            K.clear_session()
        self.version = version
        self.verbose = verbose
        self.random_seed = random_seed
        if self.verbose:
            print('Instantiation. V {}'.format(self.version))
        self.RM_type = RM_type
        self.scaling_y = scaling_y
        self.use_RobustScaler = use_RobustScaler 
        self.scaler = None
        self.scaler_y = None
        self.pca_N = pca_N
        self.pca = None
        self.train_scaled = False
        self.test_scaled = False
        self.noise = noise
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
        self.add_noise()
        self.discretized = False
        if self.N_y_bins is not None or self.y_vects is not None:
            self.discretize()
            self._init_dims(train=True, test=True)
        if scaling:
            self.scale_sets(use_log=use_log)
        self.RMs = None
        self.trained = False
        self._multi_predic = True
        if RM_filename is not None:
            self.load_RM(filename=RM_filename)
        if self.verbose:
            print('Training set size = {}, Test set size = {}'.format(self.N_train, self.N_test))
        
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
        
    def _copy_None(self, v):
        if v is None:
            to_return = None
        else:
            to_return = v.copy()
            if np.ndim(v) == 1:
                to_return = np.expand_dims(to_return, axis=1)
        return to_return
    
    def init_RM(self, **kwargs):
        """
        Initialisation of the Regression Model.
        Any parameter is passed to the Model.
        self.N_out RM can be needed if not ANN type. 
        They are stored in self.RMs list.
        """
        self.RMs = []
        self.train_params = {}
        if self.RM_type in ('ANN', 'ANN_Dis'):
            self.RMs = [MLPRegressor(**kwargs)]
            self._multi_predic = True
        elif self.RM_type == 'SVM':
            for i in range(self.N_out):
                self.RMs.append(SVR(**kwargs))
            self._multi_predic = False
        elif self.RM_type == 'NuSVM':
            for i in range(self.N_out):
                self.RMs.append(NuSVR(**kwargs))
            self._multi_predic = False
        elif self.RM_type == 'BR':
            for i in range(self.N_out):
                self.RMs.append(BayesianRidge(**kwargs))
            self._multi_predic = False
        elif self.RM_type == 'AB':
            for i in range(self.N_out):
                self.RMs.append(AdaBoostRegressor(**kwargs))
            self._multi_predic = False
        elif "Keras" in self.RM_type:
            if not TF_OK:
                raise ValueError('Tensorflow not installed, Keras RM_type not available')
            def get_kwargs(kw, default):
                if kw in kwargs:
                    return kwargs[kw]
                else:
                    return default
            activation = get_kwargs('activation', 'relu')
            kernel_initializer = get_kwargs('kernel_initializer', 'normal')
            optimizer = get_kwargs('optimizer', get_kwargs('solver', 'adam'))
            epochs = get_kwargs('epochs', 1)
            batch_size = get_kwargs('batch_size', None)
            validation_split = get_kwargs('validation_split', 0.0)
            hidden_layer_sizes = get_kwargs('hidden_layer_sizes', (10,10))
            random_state = get_kwargs('random_state', None)
            dropout = get_kwargs('dropout', None)
            tf.set_random_seed(random_state)
            model = Sequential()
            model.add(Dense(hidden_layer_sizes[0], 
                            input_dim=self.N_in, 
                            kernel_initializer=kernel_initializer, 
                            activation=activation))
            for hidden_layer_size in hidden_layer_sizes[1:]:
                model.add(Dense(hidden_layer_size, 
                                activation=activation))
                if dropout is not None:
                    model.add(Dropout(dropout))
            if self.RM_type == 'Keras':
                model.add(Dense(self.N_out, 
                                activation='linear'))
                metrics = get_kwargs('metrics', ['mse','mae'])
                model.compile(loss='mse', 
                              optimizer=optimizer, 
                              metrics=metrics)
                self.RMs = [model]
                self.train_params = {'epochs': epochs, 
                                     'batch_size': batch_size, 
                                     'verbose': False, 
                                     'validation_split': validation_split}
                self._multi_predic = True
            elif self.RM_type == 'KerasDis':
                model.add(Dense(self.N_out, 
                                activation='softmax'))
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
        else:
            raise ValueError('Unkown Regression method {}'.format(self.RM_type))
        if self.verbose:
            print('Regression Model {}'.format(self.RM_type))

    def add_noise(self, only_test=False):
        if self.noise is not None:
            if not only_test:
                noise = np.random.normal(0.0, self.noise, self.X_train.shape)
                self.X_train *= 1 + noise
            if self.X_test is not None:
                noise = np.random.normal(0.0, self.noise, self.X_test.shape)
                self.X_test *= 1 + noise
            if self.verbose:
                print("Adding noise {}".format(self.noise))
            
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
                self.y_vects = np.linspace(self.minmax[0], self.minmax[1], self.N_y_bins)
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
            if self.verbose:
                print("Discretizing column {} on {} bins".format(i, self.N_y_bins[i])) 
        return new_y
            
    def set_train(self, X=None, y=None, scaleit=False, use_log=False):
        
        self.X_train = self._copy_None(X)
        self.y_train = self._copy_None(y)
        self.train_scaled = False
        self.discretized = False
        if self.N_y_bins is not None or self.y_vects is not None:
            self.discretize()
            self._init_dims(train=True, test=True)
        if scaleit:
            self.scale_sets(use_log=use_log)
        else:
            self.scaler = None
        self._init_dims(train=True, test=False)

    def set_test(self, X=None, y=None, scaleit=False, use_log=False):

        self.X_test = self._copy_None(X)
        self.y_test = self._copy_None(y)
        self.add_noise(only_test=True)
        self.test_scaled = False
        if scaleit:
            self.scale_sets(use_log=use_log)
        self.y_test_ori = self._copy_None(self.y_test)
        self._init_dims(train=False, test=True)
        if self.N_y_bins is not None or self.y_vects is not None:
            self.y_test = self._discretize1(self.y_test)
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
            X = np.log10(X)
            isfin = np.isfinite(X).sum(1) == n_keys
            X = X[isfin]
            if y is not None:
                y = y[isfin]
            
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
            self.pca.fit(self.X_train)

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
        if (not self.train_scaled) or force:
            self.X_train_unscaled = self._copy_None(self.X_train)
            if use_log:
                self.X_train, self.y_train = self._log_data(self.X_train, self.y_train)
                log_str = 'Log10 applied. '
            else:
                log_str = ''
            pca_str = ''
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
                print('{}{}Train data scaled.'.format(log_str,pca_str))
                
        if (not self.test_scaled) or force:
            if self.X_test is not None:
                self.X_test_unscaled = self._copy_None(self.X_test)
                if use_log:
                    self.X_test, self.y_test = self._log_data(self.X_test, self.y_test)
                    log_str = 'Log10 applied. '
                else:
                    log_str = ''
                if self.X_test is not None:
                    self.X_test = self.scaler.transform(self.X_test)
                    if self.pca is not None:
                        self.X_test = self.pca.transform(self.X_test)
                        pca_str = 'PCA {} components applied. '.format(self.pca_N)
                    else:
                        pca_str = ''    
                    # No need to scale y_test as pred is inverse scaled after determined.
                    #if self.scaling_y and self.scaler_y is not None and self.y_test is not None:
                    #    self.y_test = self.scaler_y.transform(self.y_test)
                    self._init_dims(train=False, test=True)
                    self.test_scaled = True
            
            if self.verbose:
                print('{}{}Test data scaled.'.format(log_str,pca_str))
                
        if self.verbose:
            print('Training set size = {}, Test set size = {}'.format(self.N_train, self.N_test))

    def train_RM(self):
        """
        Training the models.
        """
        start = time.time()
        if not self.train_scaled and self.verbose:
            print('WARNING: training data not scaled')
        self.train_score = []
        if self.N_train != self.N_train_y:
            raise Exception('N_train {} != N_train_y {}'.format(self.N_train,
                            self.N_train_y))
        if self.verbose:
            print('Training {} inputs for {} outputs with {} data'.format(self.N_in,
                  self.N_out, self.N_train_y))
        if self._multi_predic:
            RM = self.RMs[0]
            RM.fit(self.X_train, self.y_train, **self.train_params)
            try:
                train_score = RM.score(self.X_train, self.y_train)
            except:
                train_score = np.nan
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
            else:
                y_trains = self.y_train.T
            for RM, y_train in zip(self.RMs, y_trains):
                RM.fit(self.X_train, y_train, **self.train_params)
                try:
                    train_score = RM.score(self.X_train, y_train)
                except:
                    train_score = np.nan
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
        

    def predict(self, scoring=True, reduce_by=None):
        """
        Compute the prediction usinf self.X_test
        Results are stored into self.pred
        if scoring, a score is computed comparing with self.y_test
        """
        start = time.time()
        if self.RMs is None:
            raise Exception('WARNING: Regression Model not set up')
        if not self.trained:
            raise Exception('WARNING: Regression Model not trained')
        if not self.test_scaled:
            print('WARNING: test data not scaled')
        if self._multi_predic:
            self.pred = self.RMs[0].predict(self.X_test)
        else:
            self.pred = []
            for RM in self.RMs:
                self.pred.append(RM.predict(self.X_test))
            self.pred = np.array(self.pred).T
        if scoring:
            if self.N_test != self.N_test_y:
                raise Exception('N_test {} != N_test_y {}'.format(self.N_test, self.N_test_y))
            if self._multi_predic:
                self.predic_score = [RM.score(self.X_test, self.y_test) for RM in self.RMs]
            else:
                if self.y_train.ndim == 1:
                    y_tests = (self.y_test,)
                else:
                    y_tests = self.y_test.T

                self.predic_score = [RM.score(self.X_test, y_test) for RM, y_test in zip(self.RMs, y_tests)]
            if self.N_out != self.N_out_test:
                raise Exception('N_out {} != N_out_test {}'.format(self.N_out,
                                self.N_out_test))
            if self.verbose:
                print('Score = {}'.format(', '.join(['{:.3f}'.format(ts) for ts in self.predic_score])))
        if self.scaling_y:
            self.pred = self.scaler_y.inverse_transform(self.pred)
        if reduce_by is not None:
            if self.N_y_bins is None:
                raise Exception('Can not reduce if N_y_bins is not defined.')
            self._norm_pred()
            if reduce_by == 'mean':
                if len(self.N_y_bins) == 1:
                    self.pred = np.dot(self.pred_norm,self.y_vects)
                else:
                    self.pred = np.zeros((self.N_test, len(self.N_y_bins)))
                    for i in np.arange(len(self.N_y_bins)):
                        if i == 0:
                            i_inf = 0
                        else:
                            i_inf = self.N_y_bins.cumsum()[i-1]
                        i_sup = self.N_y_bins.cumsum()[i]
                        self.pred[:,i] = np.dot(self.pred_norm[:,i_inf:i_sup],self.y_vects[i])
            elif reduce_by == 'max':
                if len(self.N_y_bins) == 1:
                    self.pred = self.y_vects[np.argmax(self.pred_norm, 1)]
                else:
                    self.pred = np.zeros((self.N_test, len(self.N_y_bins)))
                    for i in np.arange(len(self.N_y_bins)):
                        if i == 0:
                            i_inf = 0
                        else:
                            i_inf = self.N_y_bins.cumsum()[i-1]
                        i_sup = self.N_y_bins.cumsum()[i]
                        self.pred[:,i] = self.y_vects[i, np.argmax(self.pred_norm[:,i_inf:i_sup], 1)]
        end = time.time()
        if self.verbose:
            print('Predicting from {} inputs to {} outputs using {} data in {:.2f} secs.'.format(self.N_in_test,
                  self.N_out, self.N_test, end - start))
        
    def save_RM(self, filename='RM_jl.sav', save_train=False, save_test=False):
        """
        Save the following values:
        self.RM_type, self.version, self.RMs, 
        self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
        self.N_in, self.N_out, self.N_in_test, self.N_out_test,
        self.N_test, self.N_test_y, self.N_train, self.N_train_y,
        self.pca_N, self.pca. self.training_time
        """
        
        if not self.trained:
            raise Exception('Regression Model not trained')
        if save_train:
            X_train, y_train = self.X_train, self.y_train
        else:
            X_train, y_train = None, None
        if save_test:
            X_test, y_test = self.X_test, self.y_test
        else:
            X_test, y_test = None, None
            
        joblib.dump((self.RM_type, self.version, self.RMs, 
                     self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                     self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                     self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                     self.pca_N, self.pca, self.training_time, self.random_seed,
                     self.noise, X_train, y_train, X_test, y_test,
                     self.train_scaled, self.test_scaled, self.verbose), filename)
        
        if self.verbose:
            print('RM save to {}'.format(filename))
        
            
    def load_RM(self, filename='RM_jl.sav'):
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
        
        RM_tuple = joblib.load(filename)
        if self.version != RM_tuple[1]:
            print('WARNING: version loaded from {} is {}. Version from RM class is {}.'.format(filename, 
                                                                  RM_tuple[1],self.version))
        if RM_tuple[1] in ("0.15"):
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                 self.pca_N, self.pca, self.training_time, self.random_seed, 
                 self.noise, self.X_train, self.y_train, self.X_test, self.y_test,
                 self.train_scaled, self.test_scaled, self.verbose) = RM_tuple
        elif RM_tuple[1] in ("0.14"):
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                 self.pca_N, self.pca, self.training_time, self.random_seed, self.noise) = RM_tuple
        elif RM_tuple[1] in ("0.12", "0.13"):
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                 self.pca_N, self.pca, self.training_time, self.random_seed) = RM_tuple
        elif RM_tuple[1] == "0.11":
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                 self.pca_N, self.pca, self.training_time) = RM_tuple
        elif RM_tuple[1] == "0.10":
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y,
                 self.pca_N, self.pca) = RM_tuple
        elif "{:.1f}".format(RM_tuple[1]) == "0.9":
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.scaler_y, self.scaling_y, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y) = RM_tuple
        elif "{:.1f}".format(RM_tuple[1]) == "0.8":
            (self.RM_type, self.version, self.RMs, 
                 self.scaler, self.train_score, self._multi_predic,
                 self.N_in, self.N_out, self.N_in_test, self.N_out_test,
                 self.N_test, self.N_test_y, self.N_train, self.N_train_y) = RM_tuple
        self.trained = True
        if self.verbose:
            print('RM loaded from {}'.format(filename))

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
                        N_y_bins=N_y_bins, y_vects=y_vects)
        RM1.init_RM(max_iter=200000, tol=0.00001, solver='lbfgs', activation='tanh',hidden_layer_sizes=(10))
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
#%%
if __name__ == "__main__":
    pass
    #RM11, RM12, RM13, RM14 = test(func='x24')
    #RM = test_x2_K()
    #RM1, RM2, RM3 = test_x2()
    
    