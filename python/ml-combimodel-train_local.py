#!/home/srvx11/lehre/users/a1254888/.conda/envs/ml_flood/bin/python
# coding: utf-8

import os, warnings
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dask
#from dask.distributed import Client, progress
#client = Client(memory_limit='100GB', processes=True)
import dask.multiprocessing
dask.config.set(scheduler='processes')
#dask.config.set(scheduler='synchronous')

import xarray as xr
from dask.diagnostics import ProgressBar
#from joblib import Parallel
os.environ['KERAS_BACKEND'] = 'theano'
import keras

# To import our custom packages, we are setting the PYTHONPATH in `link_src`.

# In[2]:


import link_src
from python.aux.utils import open_data
from python.aux.ml_flood_config import path_to_data


era5 = open_data(path_to_data+'danube/', kw='era5')
glofas = open_data(path_to_data+'danube/', kw='glofas_ra')

#era5 = era5['tp'].isel(time=slice(6860,6990))
#print(era5.time.values)
# era5.to_netcdf('test.nc')
# print(era5.time.to_pandas().index.is_unique)
# era5b = xr.open_dataset('test.nc')
# print(era5b.time.to_pandas().index.is_unique)


## spatial subset for quick testing
era5 = era5.sel(longitude=slice(0., 14.))
glofas = glofas.sel(longitude=slice(0., 14.))



if 'tp' in era5:
    era5['tp'] = era5['tp']*1000
else:
    era5['tp'] = (era5['cp']+era5['lsp'])*1000

# no interpolation necessary

era5['reltop'] = era5['z'].sel(level=500) - era5['z'].sel(level=850)

era5['q_mean'] = era5['q'].mean('level')


# ## Prepare training data for the LocalModel
# The predictors and predictand are as follows:
# predictors = ERA5 variables of the last three days (t-1, t-2, t-3)
# predicand = (reanalysis - transport_model_forecast) (t)
#
# To achieve this, we add the variables three times as new variable to the training data `X` (with an additional ´-1´ to show that they are shifted by one for example). Each time the data is shifted by one day. Finally we drop the variables at the current day $t$ so that we are left with 3 variables, e.g. discharges at (t-1, t-2, t-3).
#
# # check
#
# The predictand variable will be create for each gridpoint just before training the local model because we have to open the transport model from disk, use it to predict the mean flow and then train the local model on the difference.

# In[6]:


listofpredictors = ['reltop', 'q_mean', 'tp', 'ro']
X_local = era5[listofpredictors]

shifts = range(1,4)
#X = add_shifted_predictors(glofas, shifts, variables='all')
#X = X.drop('dis')  # current dis is to be predicted, is not a feature


# In[7]:


X_local


# ### additionally
# we need the discharge of the last days for the transport model forecast

# In[8]:


from python.aux.utils_flowmodel import add_shifted_predictors


# In[9]:


shifts = range(1,4)
X_flow = add_shifted_predictors(glofas, shifts, variables='all')
X_flow = X_flow.drop('dis')  # current dis is to be predicted, is not a feature


# Next, we select the training and validation periods,

# In[10]:
debug = False

N_train = dict(time=slice(None, '1990'))
N_valid = dict(time=slice('1990', '1995'))


# We determine the location to save the model object and plots about the quality of training and the used features in space (the upstream river gridpoints).

# In[11]:


# kind, lat, lon will be replaced!
main_dir = '/home/srvx11/lehre/users/a1254888/ipython/ml_flood/'
ff_mod = main_dir+'/models/localmodel/danube/kind/point_lat_lon_localmodel.pkl'
ff_hist = main_dir+'/models/localmodel/danube/kind/point_lat_lon_history.png'
ff_valid = main_dir+'/models/localmodel/danube/kind/point_lat_lon_validation.png'


ff_mod_transport = main_dir+'/models/flowmodel/danube/kind/point_lat_lon_flowmodel.pkl'


# ### Model selection
# Now we can select which model to use for the LocalModel.
#
# We choose the neural network here, where scaling is done within the model, but we could use other models with feature scaling and feature selection (principal component analysis) too.

# In[34]:


from python.aux.floodmodels import LocalModel, FlowModel


# In[45]:




# ### Spatial feature selection
# In contrast to the transport model we have no background info which gridpoints influence the predictand the most, so we use dimensionality reduction approach and/or let the model decide which gridpoints are most relevant.
#
# The only hard constraint for the LocalModel is the influence radius of 1.5 degrees latitude/longitude, about 170 km and that the gridpoints have to lie within the catchment basin of the point.

# In[46]:


from python.aux.utils_flowmodel import get_mask_of_basin


# In[47]:

map = xr.ones_like(glofas['dis'].isel(time=0).drop('time'))
mask_catchment = get_mask_of_basin(map, 'Danube')

if debug:
    plt.imshow(mask_catchment.astype(int))
    plt.title('Catchment basin of the Danube river')
    plt.show()


# In[48]:


def select_riverpoints(dis):
    return (dis > 10) #.drop('time')


# In[49]:


dis_map_mean = glofas['dis'].mean('time')
is_river = select_riverpoints(dis_map_mean)

mask_river_in_catchment = is_river & mask_catchment


if debug:
    plt.imshow(mask_river_in_catchment.astype(int))
    plt.title('mask_river_in_catchment')
    plt.show()


# ## Preparing the training tasks to run in parallel
# Before we can start training all the gridpoints models in parallel, we have to prepare all the work in a list which is then worked off.
#
# We iterate over the array of river gridpoints, where the localmodel shall contribute to discharge and
# append a `delayed` function call to the `task_list`, which is worked off later on by `joblib`'s `Parallel()` function.
#
# Within the `train_localmodel()`, the TransportModel is called to predict the background flow, then the LocalModel is trained on the residual (error) of the TransportModel.
#
# The two models can then be run 14 times to produce a 14 day forecast.
# From a physics point of view, the combination model uses GloFAS discharge as initial conditions and ERA5 as boundary conditions.

# In[53]:

#pipe = Pipeline([('scaler', StandardScaler()),
#                 #('pca', PCA(n_components=6)),
#                 ('model', FlowModel('Ridge', dict(alphas=np.logspace(-3, 2, 6)))),])

#model = LocalModel('neural_net', dict(epochs=1000,))


#warnings.filterwarnings('ignore')
from joblib import Parallel, delayed  #  parallel computation
from joblib import dump, load   # saving and loading pipeline objects ("models")
from sklearn.base import clone
from python.aux.utils_flowmodel import select_upstream, preprocess_reshape_flowmodel

@dask.delayed
def train_localmodel(X_local, X_flow,
                     pipelist_local,
                     #mask_river_in_catchment, glofas, era5,
                     lat, lon,
                     f_mod_local, f_hist, f_valid,
                     f_mod_transport,
                     debug=False):
    """Train the local model for one gridpoint
    & save it to disk.
    """

    mask_river_in_catchment = xr.open_dataset('mask_river_in_catchment.nc')['mask']
    glofas = xr.open_dataset('glofas.nc')
    era5 = xr.open_dataset('era5.nc')
    #import IPython; IPython.embed()
    #print()
    #print(glofas.time, era5.time)

    upstream = select_upstream(mask_river_in_catchment, lat, lon, basin='Danube')
    N_upstream = int(upstream.sum())
    lats, lons = str(lat), str(lon)

    if not os.path.isfile(f_mod_transport) or N_upstream <= 5:
        if debug:
            print(lats, lons, 'is spring.')  # assume constant discharge
            y_flow = glofas['dis'].sel(latitude=lat, longitude=lon).mean('time')
    else:
        dis_point = glofas['dis'].sel(latitude=float(lat), longitude=float(lon))
        tp_box = era5['tp'].sel(latitude=slice(lat+1.5, lat-1.5),
                                longitude=slice(lon-1.5, lon+1.5))
        hasprecip = tp_box.mean(['longitude', 'latitude']) > 0.5

        if debug:
            print('predict mean flow using the transport model...')
            print('upstream:', N_upstream)

        # prepare the transport model input data
        Xt = X_flow.where(upstream)  # &
        Xt = Xt.where(hasprecip, drop=True)
        if debug:
            plt.imshow(upstream.astype(int))
            plt.title('upstream')
            plt.show()

        yt = dis_point
        if debug:
            print('preprocessing for the flowmodel...')
        Xda, yda, time = preprocess_reshape_flowmodel(Xt, yt)
        X_flow = Xda
        if debug:
            print(X_flow.shape)

        ppipe = load(f_mod_transport)
        y_flow = ppipe.predict(X_flow)
        # background forecast finished, calculate residual
        y_res = dis_point - y_flow

        """train local model"""
        Xt = X_local.sel(latitude=slice(lat+1.5, lat-1.5),
                          longitude=slice(lon-1.5, lon+1.5))
        Xt = Xt.where(hasprecip)
        # cluster by distance

        yt = y_res
        Xda, yda, time = preprocess_reshape_flowmodel(Xt, yt)

        X_train = Xda.loc[N_train]
        y_train = yda.loc[N_train]
        X_valid = Xda.loc[N_valid]
        y_valid = yda.loc[N_valid]

        if debug:
            print(X_train.shape, y_train.shape)
            print(X_valid.shape, y_valid.shape)

        pipe_local = Pipeline(pipelist_local)  # copy the input pipe
        is_kerasmodel = isinstance(pipe_local.named_steps['model'].m, keras.models.Sequential)
        uses_pca = 'pca' in pipe_local.named_steps

        if uses_pca and is_kerasmodel:  # need to prepare the validation data
            X_valid_np = pipe_local.named_steps['pca'].fit_transform(X_valid.values)
        else:
            X_valid_np = X_valid.values

        if debug:
            print('training the localmodel ...')

        if is_kerasmodel:
            history = pipe_local.fit(X_train.values, y_train.values,
                               model__validation_data=(X_valid_np,
                                                       y_valid.values))
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
        else:  # not keras
            pipe_local.fit(X_train.values, y_train.values)

        dump(pipe_local, f_mod)  # save it
        if debug:
            print(f_mod, 'saved.')

        # predict the validation period with it
        # 1) the residual and 2) the total with the floodmodel
        #ppipe = load(f_mod)

        try:
            y_res = pipe_local.predict(X_valid)

            fig, ax = plt.subplots(1,1, figsize=(10, 4))
            y_valid.plot(ax=ax, label='reanalysis')
            y_res.plot(ax=ax, label='prediction')
            ax.legend()
            plt.title('validation period')
            rmse = float(((y_res-y_valid)**2).to_pandas().mean()**.5)
            ax.set_title(r'residual prediction - RMSE: '+str(round(rmse,1))+' m$^3$/s')
            fig.savefig(f_valid); plt.close('all')
        except Exception as e:
            raise #warnings.warn(str(e))


# In[54]:


def mkdir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def replace(string: str, old_new: dict):
    for o, n in old_new.items():
        string = string.replace(o, str(n))
    return string


# In[55]:
# persist data in memory
mask_river_in_catchment = mask_river_in_catchment.load()
glofas = glofas.load()
era5 = era5.load()

mask_river_in_catchment.name = 'mask'
mask_river_in_catchment.to_netcdf('mask_river_in_catchment.nc')
glofas.to_netcdf('glofas.nc')
era5.to_netcdf('era5.nc')


flowmodel_kind = 'neural_net'  #= FlowModel('neural_net', dict(epochs=1000,))
localmodel = LocalModel('adaboost', dict())

pipelist_local = [#('pca', PCA(n_components=20)),
                  ('model', localmodel),]

mkdir(os.path.dirname(ff_mod).replace('kind', localmodel.kind))
task_list = []
i = 0
for ilon in range(len(mask_river_in_catchment.longitude)):
    for ilat in range(len(mask_river_in_catchment.latitude)):
        point = mask_river_in_catchment[ilat, ilon]
        if point:   # valid danube river point


            lat, lon = float(point.latitude), float(point.longitude)
            #lat, lon = 48.35, 15.650000000000034

            f_mod = replace(ff_mod, dict(lat=lat, lon=lon, kind=localmodel.kind))
            f_hist = replace(ff_hist, dict(lat=lat, lon=lon, kind=localmodel.kind))
            f_valid = replace(ff_valid, dict(lat=lat, lon=lon, kind=localmodel.kind))

            f_mod_transport = replace(ff_mod_transport, dict(lat=lat, lon=lon, kind=flowmodel_kind))

            transport_exists = os.path.isfile(f_mod_transport)
            localmodel_exists = os.path.isfile(f_mod)

            if transport_exists and not localmodel_exists:
                task = train_localmodel(X_local, X_flow,
                                     pipelist_local,
                                     #mask_river_in_catchment, glofas, era5,
                                     lat, lon,
                                     f_mod, f_hist, f_valid,
                                     f_mod_transport,
                                     debug=False)
                task_list.append(task)
                i+=1


print('number of tasks:', len(task_list))


# ### Actual computation
# The actual computation can be done for example with these two packages. We found `joblib` to work easier out of the box because we have independent computations.

# 1. Import `dask`
# 2. Decorate the function to parallelize with `@dask.delayed`
# 3. Call
with ProgressBar():
    dask.compute(task_list)
#progress(dask.compute(task_list))


# joblib:
# 1. Import `from joblib import delayed, Parallel`
# 2. Decorate the function to parallelize with `@delayed`
# 3. Call
#``Parallel(n_jobs=20, verbose=10)(task_list)``

# In[22]:


#Parallel(n_jobs=1, verbose=10)(task_list)


# The work is now finished and saved in our folder.

# In[23]:


files = os.listdir(os.path.dirname(f_mod))
len(files)


# Let's look at some of the results:

# In[24]:




# In[ ]:





# In[118]:


# lat, lon = 48.35, 15.650000000000034
# f_mod_transport = replace(ff_mod_transport, dict(lat=lat, lon=lon, kind=flowmodel.kind))
#
#
#
# upstream = select_upstream(mask_river_in_catchment, lat, lon, basin='Danube')
# N_upstream = int(upstream.sum())
# lats, lons = str(lat), str(lon)
#
# if not os.path.isfile(f_mod_transport) or N_upstream <= 5:
#     if debug:
#         print(lats, lons, 'is spring.')  # assume constant discharge
#         y_flow = glofas['dis'].sel(latitude=lat, longitude=lon).mean('time')
# else:
#     dis_point = glofas['dis'].sel(latitude=float(lat), longitude=float(lon))
#     tp_box = era5['tp'].sel(latitude=slice(lat+1.5, lat-1.5),
#                             longitude=slice(lon-1.5, lon+1.5))
#     hasprecip = tp_box.mean(['longitude', 'latitude']) > 0.5
#
#     if debug:
#         print('predict mean flow using the transport model...')
#         print('upstream:', N_upstream)


# In[ ]:





# In[ ]:





# In[ ]:


# add the data to the existing dataset
for arr_name, arr in zip(array_names, res):
    ds[arr_name] = arr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
