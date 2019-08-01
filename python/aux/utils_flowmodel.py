"""
contains methods for the flowmodel (transport model & local model)
"""
import os
from os.path import join as pjoin
from os.path import isfile
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
import xarray as xr
import geopandas
from rasterio import features
from affine import Affine


def get_mask_of_basin(da, kw_basins='Danube'):
    """Return a mask where all points outside the selected basin are False.
    
    Parameters:
    -----------
        da : xr.DataArray
            contains the coordinates
        kw_basins : str
            identifier of the basin in the basins dataset
    """
    def transform_from_latlon(lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def rasterize(shapes, coords, fill=np.nan, **kwargs):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xray coordinates. This only works for 1d latitude and longitude
        arrays.
        """
        transform = transform_from_latlon(coords['latitude'], coords['longitude'])
        out_shape = (len(coords['latitude']), len(coords['longitude']))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, **kwargs)
        return xr.DataArray(raster, coords=coords, dims=('latitude', 'longitude'))
    
    # this shapefile is from natural earth data
    # http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/
    shp2 = '/raid/home/srvx7/lehre/users/a1303583/ipython/ml_flood/data/drainage_basins/Major_Basins_of_the_World.shp'
    basins = geopandas.read_file(shp2)
    single_basin = basins.query("NAME == '"+kw_basins+"'").reset_index(drop=True)
    shapes = [(shape, n) for n, shape in enumerate(single_basin.geometry)]

    da['basins'] = rasterize(shapes, da.coords)
    da = da.basins == 0
    return da.drop('basins')  # the basins coordinate is not used anymore from here on

def select_upstream(mask_river_in_catchment, lat, lon, basin='Danube'):
    """Return a mask containing upstream river gridpoints.
    
    Parameters
    ----------
        mask_river_in_catchment : xr.DataArray
            array that is True only for river gridpoints within a certain catchment
            coords: only latitude and longitute
    
        lat, lon : float
            latitude and longitude of the considered point
            
        basin : str
            identifier of the basin in the basins dataset
    """
    
    # this condition should be replaced with a terrain dependent mask
    # but generally speaking, there will always be some points returned that
    # do not influence the downstream point; 
    # the statistical model should ignore those points as learned from the dataset
    da = mask_river_in_catchment.load()
    is_west = (~np.isnan(da.where(da.longitude <= lon))).astype(bool)
    
    mask_basin = get_mask_of_basin(da, kw_basins=basin)

    nearby_mask = da*0.
    nearby_mask.loc[dict(latitude=slice(lat+1.5, lat-1.5), 
                         longitude=slice(lon-1.5, lon+1.5))] = 1.
    nearby_mask = nearby_mask.astype(bool)
    
    mask = mask_basin & nearby_mask & is_west & mask_river_in_catchment
    
    if 'basins' in mask.coords:
        mask = mask.drop('basins')
    if 'time' in mask.coords:
        mask = mask.drop('time')  # time and basins dimension make no sense here
    return mask

def add_shifted_predictors(ds, shifts, variables='all'):
    """Adds additional variables to an array which are shifted in time.
    
    Parameters
    ----------
        ds : xr.Dataset
        shifts : list of integers
        variables : str or list
    """
    if variables == 'all': 
        variables = ds.data_vars
        
    for var in variables:
        for i in shifts:
            if i == 0: continue  # makes no sense to shift by zero
            newvar = var+'-'+str(i)
            ds[newvar] = ds[var].shift(time=i)
    return ds

def preprocess_reshape_flowmodel(X_dis, y_dis):
    """Reshape, merge predictor/predictand in time, drop nans.

    Parameters
    ----------
        X_dis : xr.Dataset
            variables: time shifted predictors (name irrelevant)
            coords: time, latitude, longitude
        y_dis : xr.DataArray
            coords: time, latitude, longitude
    """
    X_dis = X_dis.to_array(dim='time_feature')  # assumes that X_dis contains time shifted predictors
    X_dis = X_dis.stack(features=['latitude', 'longitude', 'time_feature'])  # seen as one dimension for the model
    Xar = X_dis.dropna('features', how='all')
    
    yar = y_dis
    yar = yar.drop(['latitude', 'longitude'])
    yar.coords['features'] = 'dis'

    Xy = xr.concat([Xar, yar], dim='features')
    Xyt = Xy.dropna('time', how='any')  # drop them as we cannot train on nan values
    time = Xyt.time
    
    Xda = Xyt[:,:-1]
    yda = Xyt[:,-1]
    return Xda, yda, time