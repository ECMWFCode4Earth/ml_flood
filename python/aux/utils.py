import os
from os.path import join as pjoin
from os.path import isfile as isfile
import numpy as np
import pandas as pd
import xarray as xr
"""
Contains various utility methods
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
    """Calculates the first two statistical moments and
    the coefficient of variation in the specified dimension.

    Parameters:
    -----------
        ds : xr.Dataset
        dim_aggregator : str
            coordinate to calculate the statistical moments over
        time_constraint : str
            longitude

    Returns
    -------
    xr.DataArray
        covariance array
    """
    if dim_aggregator == 'spatial':
        dim_aggregator = ['latitude', 'longitude']
    else:
        dim_aggregator = 'time'

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


def spatial_cov(da, lat=48, lon=15):
    """Calculates the spatial (auto)-covariance for the specified point (lat, lon)
    under the specified time constraint.

    Parameters:
    -----------
        da : xr.DataArray
            contains the 3-dimensional array (time, latitude, longitude)
        lat : float
            latitude
        lon : float
            longitude

    Returns
    -------
    xr.DataArray
        covariance array
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
    """Calculates the spatial covariance between `da_point` and `da`.

    Parameters
    ----------
    da_point : xr.DataArray
        contains timeseries data (no spatial coordinates)
    da : xr.DataArray
        contains the 3-dimensional array (time, latitude, longitude)

    Returns
    -------
    xr.DataArray
        covariance array
    """
    import numpy as np
    if not isinstance(da, xr.core.dataarray.DataArray):
        print('data input has to be a xarray data array')
    # set nans to zero, or else the dot function will return nans
    da = da.where(~np.isnan(da), 0)
    da_point = da_point.where(~np.isnan(da_point), 0)
    da = da.load()
    da_point = da_point.load()
    anomalies = (da - da.mean('time'))
    anomalies_point = (da_point - da_point.mean('time'))
    scal_prod = (anomalies_point.dot(anomalies)).compute()
    stds = (da_point.std(dim='time')*da.std(dim='time')).compute()
    total_num = da.time.shape[0]
    return scal_prod/stds/total_num


def open_data(path, kw='era5'):
    """
    Opens all available ERA5/glofas datasets (depending on the keyword) in
    the specified path and resamples time to match the timestamp /per day (through
    the use of cdo YYYYMMDD 23z is the corresponding timestamp) in the case of era5,
    or renames lat lon in the case of glofas.
    """
    combine = 'by_coords'
    if kw == 'era5':
        ds = xr.open_dataset(path+'era5_danube_pressure_and_single_levels.nc')
        static = xr.open_dataset(path+'era5_slt_z_slor_lsm_stationary_field.nc')
        static = static.isel(time=0).drop('time')
        static = static.rename(dict(z='z_topo'))
        ds = xr.merge([ds, static])
    elif kw == 'glofas_ra':
        ds = xr.open_mfdataset(path+'*glofas_reanalysis*', combine=combine)
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif kw == 'glofas_fr':
        ds = xr.open_mfdataset(path+'*glofas_forecast*', combine=combine)
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    return ds


def shift_time(ds, value):
    """Shift the time coordinate, i.e. add a certain number of hours
    to the coordinate values.
    """
    ds.coords['time'].values = pd.to_datetime(ds.coords['time'].values) + value
    return ds


def calc_area(da, resolution_degrees=None):
    """Calculate the area for each gridpoint of a 2-dimensional DataArray.

    Approximations: spherical earth, gridbox is a square of area dx*dy

    Parameters
    ----------
        da : xr.DataArray
            a 2-dimensional array with coordinates `latitude` and `longitude`

        resolution_degrees : float
            grid resolution in degrees latitude/longitude, e.g. 0.25

    Returns
    -------
    xr.DataArray
        contains the area per gridbox in m^2
    """
    km_deg = 111319  # meters per degree latitude

    if not resolution_degrees:
        if (len(da.latitude) < 2 or len(da.longitude) < 2):
            raise ValueError('Either lat or lon is singleton, cannot infer'
                             ' resolution, pass `resolution_degrees` to continue!')
        res_lat = abs(da.latitude[0].values-da.latitude[1].values)
        res_lon = abs(da.longitude[0].values-da.longitude[1].values)
    else:
        res_lat = resolution_degrees
        res_lon = resolution_degrees

    lats = da.latitude.values[:, np.newaxis]*np.ones(len(da.longitude))
    dx = km_deg*abs(np.cos(lats/90))*res_lon
    dy = km_deg*res_lat
    for var in da:
        area = dx*dy
        area = xr.DataArray(area, dims=['latitude', 'longitude'],
                            coords=dict(latitude=('latitude', da.latitude),
                                        longitude=('longitude', da.longitude)))
        area.name = 'area'
        area.attrs['units'] = 'meters'
        return area
