import os
import numpy as np
import xarray as xr

from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
#from dask_ml.decomposition import PCA

from dask_ml.xgboost import XGBRegressor
#from dask_ml.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

import h5py
import keras
from keras.layers.core import Dropout
from keras.models import load_model


def add_time(vector, time, name=None):
    """Converts numpy arrays to xarrays with a time coordinate."""
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)

class FlowModel_DNN(object):
    def __init__(self, **kwargs):
        model = keras.models.Sequential()
        self.cfg = kwargs
        
        model.add(keras.layers.BatchNormalization())
        
        #model.add(Dropout(0.25))
        model.add(keras.layers.Dense(8,
                                  kernel_initializer=keras.initializers.Zeros(), 
                                  kernel_regularizer=keras.regularizers.l2(1e-4),
                                  bias_initializer='zeros',
                                  activation='relu'))

        model.add(keras.layers.Dense(1, activation='linear'))
        
        #opti = keras.optimizers.RMSprop(lr=.05)
        opti = keras.optimizers.Adadelta(lr=0.05, rho=0.95, epsilon=None, decay=0.0)
        #opti = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.8, nesterov=True)

        model.compile(loss='mean_squared_error', 
                      optimizer=opti)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=1, patience=100, verbose=0, mode='auto',
                            baseline=None, restore_best_weights=True),]
                         #keras.callbacks.ModelCheckpoint(self.cfg.get('filepath'), 
                         #   monitor='val_loss', verbose=0, save_best_only=True, 
                         #   save_weights_only=False, mode='auto', period=1),]

    def predict(self, Xda):
        return self.model.predict(Xda)

    def fit(self, Xda, yda, **kwargs):
        return self.model.fit(Xda, yda.reshape(-1,1),
                              epochs=self.cfg.get('epochs', None),
                              batch_size=512,
                              callbacks=self.callbacks,
                              verbose=0,
                              **kwargs)

    
class FlowModel(object):
    def __init__(self, kind, model_config):
        """Model selection & Xarray compatibility"""
        self.kind = kind
        if kind=='neural_net':
            self.m = FlowModel_DNN(**model_config)
        elif kind=='xgboost':
            self.m = XGBRegressor(**model_config)
        elif kind=='Ridge':
            self.m = RidgeCV(**model_config)
        else:
            raise NotImplementedError(str(kind)+' not defined')
    
    def fit(self, Xda, yda, **kwargs):
        return self.m.fit(Xda, yda, **kwargs)
    
    def predict(self, Xda, name=None):
        # use with xarray, return xarray
        a = self.m.predict(Xda.values).squeeze()
        return add_time(a, Xda.time, name=name)
    
#def FlowModel(kind, model_config):
#    if kind=='neural_net':
#        return FlowModel_DNN(**model_config)
#    elif kind=='xgboost':
#        return XGBRegressor(**model_config)
#    elif kind=='Ridge':
#        return RidgeCV(**model_config)
#    else:
#        raise NotImplementedError(str(kind)+' not defined')
                
                
                
                
            
    