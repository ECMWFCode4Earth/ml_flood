import os
from os.path import join as pjoin
from os.path import isfile as isfile
import xarray as xr
import pandas as pd
import datetime as datetime
"""
contains various utility methods
"""

def rename_files(path, old, new, str_constraint=None):
    """
    loops through the given directory and replaces the specified part of a filename with the new part
    the str_constrait variable can be used to only include files, which include a specified string
    """
    for name in os.listdir(path):
        if str_constraint:
            if str_constraint in name:
                if old in name:
                    name_new = name.replace(old, new)
                    print(f'renaming {name} to {name_new} in {path} ...')
                    os.rename(pjoin(path, name), pjoin(path, name_new))
        else:
            if old in name:
                name_new = name.replace(old, new)
                print(f'renaming {name} to {name_new} in {path} ...')
                os.rename(pjoin(path, name), pjoin(path, name_new))

                
def cdo_daily_means(path, file_includes):
    """
    loops through the given directory and and executes "cdo dayavg *file_includes* file_out"
    appends "dayavg" at the end of the filename
    """
    for name in os.listdir(path):
        if file_includes in name and 'dayavg' not in name:
            name_new = f"{''.join(name.split('.')[:-1])}_dayavg.{name.split('.')[-1]}"
            print(f'calculating daily means for {name} to {name_new} in {path} ...')
            os.system(f'cdo dayavg {pjoin(path, name)} {pjoin(path, name_new)}')
            

def cdo_precip_sums(path, file_includes='precipitation'):
    """
    loops through the given directory and and executes "cdo -b 32 daysum filein.nc fileout.nc"
    appends "daysum" at the end of the filename
    """
    for name in os.listdir(path):
        if file_includes in name and 'dayavg' not in name and 'daysum' not in name:
            name_new = f"{''.join(name.split('.')[:-1])}_daysum.{name.split('.')[-1]}"
            print(f'calculating daily sums for {name} to {name_new} in {path} ...')
            os.system(f'cdo -b 32 daysum {pjoin(path, name)} {pjoin(path, name_new)}')
            
            
def cdo_clean_precip(path, precip_type='precipitation'):
    """
    loops through the given directory and and executes "ncks -v cp,tp filein.nc fileout.nc" or "ncks -x -v cp,tp filein.nc fileout.nc" for all files which contain precip_type in their name and creates new files with the corresponding variables
    """
    for name in os.listdir(path):
        if precip_type in name and 'dayavg' in name:
            name_new = name.replace('total_precipitation_', '').replace('convective_precipitation_', '')
            if isfile(pjoin(path, name_new)):
                print(f'{name_new} already exists in {path}, delete it first before running again!')
            else:
                print(f'clear precipitation vars from {name} to {name_new} in {path} ...')
                os.system(f'ncks -x -v tp,cp {pjoin(path, name)} {pjoin(path, name_new)}')
        elif precip_type in name and 'daysum' in name:
            name_new = f'era5_total_precipitation_convective_precipitation_{name[-17:]}'
            if isfile(pjoin(path, name_new)):
                print(f'{name_new} already exists in {path}, delete it first before running again!')
            else:
                print(f'write precipitation vars from {name} to {name_new} in {path} ...')
                os.system(f'ncks -v tp,cp {pjoin(path, name)} {pjoin(path, name_new)}')
            

def cdo_merge_time(path, file_includes, new_file):
    """
    merges all files including a specified string in their name within the given directory into the specified new file with "cdo mergetime *file_includes* fileout.nc"
    """
    if isfile(pjoin(path, new_file)):
        print(f'{new_file} already exists in {path}, delete it first before running again!')
    else:
        print(f'merging time for files including "{file_includes}" into {new_file} in {path} ...')
        os.system(f'cdo mergetime {pjoin(path, f"*{file_includes}*")} {pjoin(path, new_file)}')


def cdo_spatial_cut(path, file_includes, new_file_includes, lonmin, lonmax, latmin, latmax):
    """
    loops through the given directory and and executes "cdo -sellonlatbox,lonmin,lonmax,latmin,latmax *file_includes* fileout.nc" appends "spatial_cut_*new_file_includes*" at the end of the filename
    """
    for name in os.listdir(path):
        if file_includes in name and 'spatial_cut' not in name:
            name_new = f"{''.join(name.split('.')[:-1])}_spatial_cut_{new_file_includes}.{name.split('.')[-1]}"
            print(f'extracting region: {name} to {name_new} in {path} ...')
            os.system(f'cdo -sellonlatbox,{lonmin},{lonmax},{latmin},{latmax} {pjoin(path, name)} {pjoin(path, name_new)}')
        

def calc_stat_moments(ds, dim_aggregator='time', time_constraint=None):
    """
    Calculates the first two statistical moments and the coefficient of variation in the specified dimension. Takes a xarray dataset as input.
    """
    if dim_aggregator == 'spatial':
        dim_aggregator = ['latitude', 'longitude']
    else:
        dim_aggregator='time'

    if time_constraint == 'seasonally':
        mu = ds.groupby('time.season').mean(dim=dim_aggregator)
        sig = ds.groupby('time.season').std(dim=dim_aggregator)
    elif time_constraint == 'monthly':
        mu = ds.groupby('time.month').mean(dim=dim_aggregator)
        sig = ds.groupby('time.month').std(dim=dim_aggregator)
    else:
        mu = ds.mean(dim=dim_aggregator)
        sig = ds.std(dim=dim_aggregator)
    vc = sig**2/mu

    ds_new = xr.concat([mu, sig, vc], dim='stat_moments')
    ds_new.coords['stat_moments'] = ['mean', 'std', 'vc']
    return ds_new


def spatial_cov(da, lat=48, lon=15, time_constraint=None):
    """
    Calculates the spatial covariance for the specified point (lat, lon) under the specified time constraint.
    """
    import numpy as np
    if not isinstance(da, xr.core.dataarray.DataArray):
        print('data input has to be a xarray data array')
    # set nans to zero, or else the dot function will return nans
    da = da.where(~np.isnan(da), 0)
    da = da.load()
    anomalies = da - da.mean('time')
    point = dict(latitude=lat, longitude=lon)
    scal_prod = (anomalies.loc[point].dot(anomalies)).compute()
    stds = (da.loc[point].std(dim='time')*da.std(dim='time')).compute()
    total_num = da.time.shape[0]
    return scal_prod/stds/total_num


def spatial_cov_2var(da_point, da):
    """
    Calculates the spatial covariance for the point series da_point and the 3D data array da.
    """
    import numpy as np
    if not isinstance(da, xr.core.dataarray.DataArray):
        print('data input has to be a xarray data array')
    # set nans to zero, or else the dot function will return nans
    da = da.where(~np.isnan(da), 0)
    da_point = da_point.where(~np.isnan(da_point), 0)
    da = da.load()
    da_point = da_point.load()
    anomalies = (da - da.mean('time'))#/da.std(dim='time')
    anomalies_point = (da_point - da_point.mean('time'))#/da_point.std(dim='time')
    scal_prod = (anomalies_point.dot(anomalies)).compute()
    stds = (da_point.std(dim='time')*da.std(dim='time')).compute()
    total_num = da.time.shape[0]
    return scal_prod/stds/total_num


def open_data(path, kw='era5'):
    """
    Opens all available ERA5/glofas datasets (depending on the keyword) in the specified path and
    resamples time to match the timestamp /per day (through the use of cdo YYYYMMDD 23z is the
    corresponding timestamp) in the case of era5, or renames lat lon in the case of glofas.
    """
    if kw is 'era5':    
        ds = xr.open_mfdataset(path+'*era5*')
        #ds.coords['time'] = pd.to_datetime(ds.coords['time'].values) - datetime.timedelta(hours=23)
    elif kw is 'glofas_ra':
        ds = xr.open_mfdataset(path+'*glofas_reanalysis*')
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif kw is 'glofas_fr':
        ds = xr.open_mfdataset(path+'*glofas_forecast*')
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    return ds




