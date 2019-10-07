import os
import warnings
import numpy as np
import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt
from dask import delayed
import xarray as xr

# from joblib import Parallel, delayed  # parallel computation
from joblib import dump, load   # saving and loading pipeline objects ("models")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import RidgeCV

import keras
from keras.layers.core import Dropout

from .utils_floodmodel import select_upstream, reshape_scalar_predictand
np.seterr(divide='ignore', invalid='ignore')


def add_time(vector, time, name=None):
    """Converts numpy arrays to xarrays with a time coordinate.

    Parameters
    ----------
    vector : np.array
        1-dimensional array of predictions
    time : xr.DataArray
        the return value of `Xda.time`

    Returns
    -------
    xr.DataArray
    """
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)


def add_time_to_sequence_output(array, time, name=None):
    """Add time coordinates to multiday model predictions.

    Parameters
    ----------
    array : numpy.array
        the prediction, 2-dimensional ('init_time', 'fxh')
    time : xr.DataArray
        the return value of `Xda.time`

    Returns
    -------
    xr.DataArray
    """
    init_time = pd.to_datetime(time.values)-dt.timedelta(hours=1)
    fxh = range(1, array.shape[1]+1)
    return xr.DataArray(array, dims=('init_time', 'fxh'),
                        coords=dict(init_time=('init_time', init_time),
                                    fxh=('fxh', fxh),
                                    name=name))


class FlowModel(object):
    """Model selection & Xarray compatibility"""
    def __init__(self, kind, model_config):
        self.kind = kind
        if kind == 'neural_net':
            self.m = FlowModel_DNN(**model_config)
        elif kind == 'xgboost':
            self.m = XGBRegressor(**model_config)
        elif kind == 'Ridge':
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

        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        model.add(keras.layers.BatchNormalization())

        # model.add(Dropout(0.25))
        model.add(keras.layers.Dense(8,
                  # kernel_initializer=keras.initializers.Zeros(),
                  # kernel_regularizer=keras.constraints.NonNeg(),
                  # bias_initializer='zeros',
                  activation='elu'))

        model.add(keras.layers.Dense(1, activation='linear'))

        # opti = keras.optimizers.RMSprop(lr=.05)
        opt = keras.optimizers.Adam()  # delta(lr=0.05, rho=0.95, epsilon=None, decay=0.0)
        # opt = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.8, nesterov=True)

        model.compile(loss='mean_squared_error',
                      optimizer=opt)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=1, patience=100, verbose=0, mode='auto',
                          baseline=None, restore_best_weights=True), ]

    def predict(self, Xda):
        X = self.xscaler.transform(Xda.values)
        y = self.model.predict(X).squeeze()
        y = self.yscaler.inverse_transform(y)
        return y

    def fit(self, X_train, y_train, validation_data=(None, None), **kwargs):
        X_valid, y_valid = validation_data

        X_train = self.xscaler.fit_transform(X_train.values)
        y_train = self.yscaler.fit_transform(y_train.values.reshape(-1, 1))

        X_valid = self.xscaler.transform(X_valid.values)
        y_valid = self.yscaler.transform(y_valid.values.reshape(-1, 1))

        return self.model.fit(X_train, y_train,
                              epochs=self.cfg.get('epochs', None),
                              batch_size=180,
                              callbacks=self.callbacks,
                              verbose=0, **kwargs)

# class FlowModel_DNN(object):
#     """Define the internals of the neural-network transport model."""
#     def __init__(self, **kwargs):
#         model = keras.models.Sequential()
#         self.cfg = kwargs
#
#         model.add(keras.layers.BatchNormalization())
#
#         # model.add(Dropout(0.25))
#         model.add(keras.layers.Dense(8,
#                   kernel_initializer=keras.initializers.Zeros(),
#                   kernel_regularizer=keras.regularizers.l2(1e-4),
#                   bias_initializer='zeros',
#                   activation='relu'))
#
#         model.add(keras.layers.Dense(1, activation='linear'))
#
#         # opti = keras.optimizers.RMSprop(lr=.05)
#         opti = keras.optimizers.Adadelta(lr=0.05, rho=0.95, epsilon=None, decay=0.0)
#         # opti = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.8, nesterov=True)
#
#         model.compile(loss='mean_squared_error',
#                       optimizer=opti)
#         self.model = model
#
#         self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
#                           min_delta=1, patience=100, verbose=0, mode='auto',
#                           baseline=None, restore_best_weights=True), ]
#
#     def predict(self, Xda):
#         return self.model.predict(Xda)
#
#     def fit(self, Xda, yda, **kwargs):
#         return self.model.fit(Xda, yda.reshape(-1, 1),
#                               epochs=self.cfg.get('epochs', None),
#                               batch_size=512,
#                               callbacks=self.callbacks,
#                               verbose=0,
#                               **kwargs)


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
        fig.savefig(f_upstream)
        plt.close('all')
        print(f_upstream)

    if debug:
        print('N_upstream', N_upstream)

    if N_upstream <= 5:
        if debug:
            print(lats, lons, 'is spring.')
    else:
        # if False: #os.path.isfile(f_mod):
        #     if debug:
        #         print('already trained.')
        # else:
        #     if debug:
        #         print(lats, lons, 'is danube river -> train flowmodel')

        try:
            fig, ax = plt.subplots()
            ax.imshow(upstream.astype(int))
            plt.title(str(N_upstream)+' upstream points for '+lats+' '+lons)
            fig.savefig(f_upstream)
            plt.close('all')
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
        Xda, yda = reshape_scalar_predictand(Xt, yt)

        X_train = Xda.loc[N_train]
        y_train = yda.loc[N_train]
        X_valid = Xda.loc[N_valid]
        y_valid = yda.loc[N_valid]

        if debug:
            print(X_train.shape, y_train.shape)
            print(X_valid.shape, y_valid.shape)
        ppipe = clone(pipe)
        history = ppipe.fit(X_train, y_train,
                            model__validation_data=(X_valid, y_valid))

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
            plt.legend()  # ['Train', 'Test'], loc='upper left')
            ax.set_yscale('log')
            fig.savefig(f_hist)
            plt.close('all')
        except Exception as e:
            warnings.warn(str(e))

        ppipe = load(f_mod)
        y_m = ppipe.predict(X_valid)

        try:
            fig, ax = plt.subplots(figsize=(10, 4))
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

        #model.add(Dropout(0.25))
        model.add(keras.layers.Dense(8,
                                  kernel_initializer=keras.initializers.Zeros(),
                                  #kernel_regularizer=keras.regularizers.l2(1e-5),
                                  bias_initializer='zeros',
                                  activation='relu'))
        model.add(keras.layers.Dense(8,
                          kernel_initializer=keras.initializers.Zeros(),
                          #kernel_regularizer=keras.regularizers.l2(1e-5),
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
        elif kind=='adaboost':
            self.m = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),
                                       n_estimators=200)
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
