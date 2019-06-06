import os
from os.path import join as pjoin
from os.path import isfile as isfile
import xarray as xr

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
            name_new = f'{name[:-3]}_dayavg.nc'
            print(f'calculating daily means for {name} to {name_new} in {path} ...')
            os.system(f'cdo dayavg {pjoin(path, name)} {pjoin(path, name_new)}')
            

def cdo_precip_sums(path, file_includes='precipitation'):
    """
    loops through the given directory and and executes "cdo -b 32 daysum filein.nc fileout.nc"
    appends "daysum" at the end of the filename
    """
    for name in os.listdir(path):
        if file_includes in name and 'dayavg' not in name and 'daysum' not in name:
            name_new = f'{name[:-3]}_daysum.nc'
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
    merges all files including a specified string in their name within the given directory into the specified new file
    """
    if isfile(pjoin(path, new_file)):
        print(f'{new_file} already exists in {path}, delete it first before running again!')
    else:
        print(f'merging time for files including "{file_includes}" into {new_file} in {path} ...')
        os.system(f'cdo mergetime {pjoin(path, f"*{file_includes}*")} {pjoin(path, new_file)}')
   

def calc_stat_moments(ds, dim_aggregator='time', time_constraint=None):
    """
    Calculates the first three statistical moments in the specified dimension. Takes a xarray dataset as input.
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
        
    ds_new = xr.concat([mu, sig], dim='stat_moments')
    ds_new.coords['stat_moments'] = ['mean', 'std']
    return ds_new