import os
import numpy as np

from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
#from dask_ml.decomposition import PCA

#from dask_ml.xgboost import XGBRegressor
#from dask_ml.linear_model import LogisticRegression
#from dask_ml.linear_model import LinearRegression
#from sklearn.linear_model import Ridge

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
                            min_delta=1, patience=10, verbose=0, mode='auto',
                            baseline=None, restore_best_weights=True),
                         keras.callbacks.ModelCheckpoint(self.cfg.get('filepath'), 
                            monitor='val_loss', verbose=0, save_best_only=True, 
                            save_weights_only=False, mode='auto', period=1),]

    def predict(self, Xda, name=None):
        a = self.model.predict(Xda.values).squeeze()
        return add_time(a, Xda.time, name=name)

    def fit(self, Xda, yda, **kwargs):
        return self.model.fit(Xda, yda.reshape(-1,1),
                              epochs=self.cfg.get('epochs', None),
                              batch_size=512,
                              callbacks=self.callbacks,
                              verbose=0,
                              **kwargs)
    
    
    
    class FlowModel(object):
        def __init__(self, kind, **kwargs):
            if kind=='NN':
                return FlowModel_DNN(**kwargs)
            elif kind=='xgboost':
                return None
            else:
                raise NotImplementedError(str(kind)+' not defined')
                
                
                
                
            
    