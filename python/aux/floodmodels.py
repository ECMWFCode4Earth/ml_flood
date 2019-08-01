import os, warnings
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import xarray as xr

from joblib import Parallel, delayed  #  parallel computation
from joblib import dump, load   # saving and loading pipeline objects ("models")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA
from dask_ml.xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV

import keras
from keras.layers.core import Dropout

from .utils_flowmodel import select_upstream, preprocess_reshape_flowmodel


def add_time(vector, time, name=None):
    """Converts numpy arrays to xarrays with a time coordinate."""
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)


class FlowModel(object):
    """Model selection & Xarray compatibility"""
    def __init__(self, kind, model_config):
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

class FlowModel_DNN(object):
    """Define the internals of the neural-network transport model."""
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


    def predict(self, Xda):
        return self.model.predict(Xda)

    def fit(self, Xda, yda, **kwargs):
        return self.model.fit(Xda, yda.reshape(-1,1),
                              epochs=self.cfg.get('epochs', None),
                              batch_size=512,
                              callbacks=self.callbacks,
                              verbose=0,
                              **kwargs)




@delayed
def train_flowmodel(X, y, pipe,
                    lat, lon,
                    tp, mask_river_in_catchment,
                    N_train, N_valid,
                    f_mod, f_hist, f_valid, f_upstream, debug=False):
    """Train the transport model, save it to disk."""

    #f_mod = replace(ff_mod, dict(lat=lat, lon=lon, kind=model.kind))
    #f_hist = replace(ff_hist, dict(lat=lat, lon=lon, kind=model.kind))
    #f_valid = replace(ff_valid, dict(lat=lat, lon=lon, kind=model.kind))
    #f_upstream = replace(ff_upstream, dict(lat=lat, lon=lon, kind=model.kind))

    upstream = select_upstream(mask_river_in_catchment, lat, lon, basin='Danube')
    N_upstream = int(upstream.sum())
    lats, lons = str(lat), str(lon)

    if N_upstream > 80:  # something wrong ?
        fig, ax = plt.subplots()
        ax.imshow(upstream.astype(int))
        plt.title(str(N_upstream)+' upstream points for '+lats+' '+lons)
        fig.savefig(f_upstream); plt.close('all')
        print(f_upstream)

    if debug:
        print('N_upstream', N_upstream)

    if N_upstream <= 5:
        if debug:
            print(lats, lons, 'is spring.')
    else:
        if os.path.isfile(f_mod):
            if debug:
                print('already trained.')
        else:
            if debug:
                print(lats, lons, 'is danube river -> train flowmodel')

            try:
                fig, ax = plt.subplots()
                ax.imshow(upstream.astype(int))
                plt.title(str(N_upstream)+' upstream points for '+lats+' '+lons)
                fig.savefig(f_upstream); plt.close('all')
            except:
                pass

            tp_box = tp.sel(latitude=slice(lat+1.5, lat-1.5),
                            longitude=slice(lon-1.5, lon+1.5))
            noprecip = tp_box.mean(['longitude', 'latitude']) < 0.1

            Xt = X.copy()
            yt = y.copy()

            Xt = Xt.where(noprecip, drop=True)
            Xt = Xt.where(upstream, drop=True)
            yt = yt.sel(latitude=float(lat), longitude=float(lon))
            Xda, yda, time = preprocess_reshape_flowmodel(Xt, yt)

            X_train = Xda.loc[N_train]
            y_train = yda.loc[N_train]
            X_valid = Xda.loc[N_valid]
            y_valid = yda.loc[N_valid]

            if debug:
                print(X_train.shape, y_train.shape)
                print(X_valid.shape, y_valid.shape)
            ppipe = clone(pipe)
            history = ppipe.fit(X_train.values, y_train.values,
                               model__validation_data=(X_valid.values,
                                                       y_valid.values))

            dump(ppipe, f_mod)

            try:
                h = history.named_steps['model'].m.model.history

                # Plot training & validation loss value
                fig, ax = plt.subplots()
                ax.plot(h.history['loss'], label='loss')
                ax.plot(h.history['val_loss'], label='val_loss')
                plt.title('Model loss')
                ax.set_ylabel('Loss')
                ax.set_xlabel('Epoch')
                plt.legend() #['Train', 'Test'], loc='upper left')
                ax.set_yscale('log')
                fig.savefig(f_hist); plt.close('all')
            except Exception as e:
                warnings.warn(str(e))

            ppipe = load(f_mod)
            y_m = ppipe.predict(X_valid)

            try:
                fig, ax = plt.subplots(figsize=(10,4))
                y_m.to_pandas().plot(ax=ax)
                y_valid.name = 'reanalysis'
                y_valid.to_pandas().plot(ax=ax)
                plt.legend()
                fig.savefig(f_valid); plt.close('all')
            except Exception as e:
                warnings.warn(str(e))

                
class LocalModel_DNN(object):
    """Define the internals of the neural-network transport model."""
    def __init__(self, **kwargs):
        model = keras.models.Sequential()
        self.cfg = kwargs

        model.add(keras.layers.BatchNormalization())

        model.add(Dropout(0.25))
        model.add(keras.layers.Dense(8,
                                  kernel_initializer=keras.initializers.Zeros(),
                                  kernel_regularizer=keras.regularizers.l2(1e-5),
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


    def predict(self, Xda):
        return self.model.predict(Xda)

    def fit(self, Xda, yda, **kwargs):
        return self.model.fit(Xda, yda.reshape(-1,1),
                              epochs=self.cfg.get('epochs', None),
                              batch_size=512,
                              callbacks=self.callbacks,
                              verbose=0,
                              **kwargs)
                
class LocalModel(object):
    """Model selection & Xarray compatibility,
    currently the same as FlowModel
    """
    def __init__(self, kind, model_config):
        self.kind = kind
        if kind=='neural_net':
            self.m = LocalModel_DNN(**model_config)
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
    
    
