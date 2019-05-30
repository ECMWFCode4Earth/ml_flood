import os
from os.path import join as pjoin

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
                    os.rename(pjoin(path, name), pjoin(path, name_new))
        else:
            if old in name:
                name_new = name.replace(old, new)
                os.rename(pjoin(path, name), pjoin(path, name_new))

                
def cdo_daily_means(path, file_includes):
    """
    loops through the given directory and and executes "cdo dayavg *file_includes* file_out"
    appends "dayavg" at the end of the filename
    """
    for name in os.listdir(path):
        if file_includes in name and 'dayavg' not in name:
            name_new = f'{name[:-3]}_dayavg.nc'
            os.system(f'cdo dayavg {pjoin(path, name)} {pjoin(path, name_new)}')
            

def cdo_precip_sums(path, file_includes='precipitation'):
    """
    loops through the given directory and and executes "cdo -b 32 daysum filein.nc fileout.nc"
    appends "daysum" at the end of the filename
    """
    for name in os.listdir(path):
        if file_includes in name and 'dayavg' not in name and 'daysum' not in name:
            name_new = f'{name[:-3]}_daysum.nc'
            os.system(f'cdo -b 32 daysum {pjoin(path, name)} {pjoin(path, name_new)}')
            
            
def cdo_clean_precip(path):
    """
    loops through the given directory and and executes "ncks -v cp,tp filein.nc fileout.nc" or "ncks -x -v cp,tp filein.nc fileout.nc" for all files which contain 'precipitation' in their name and creates new files with the corresponding variables
    """
    for name in os.listdir(path):
        if 'precipitation' in name and 'dayavg' in name:
            name_new = name.replace('total_precipitation_', '').replace('convective_precipitation_', '')
            os.system(f'ncks -x -v tp,cp {pjoin(path, name)} {pjoin(path, name_new)}')
        elif 'precipitation' in name and 'daysum' in name:
            name_new = f'{name[:5]}total_precipitation_convective_precipitation_{name[-17:]}'
            os.system(f'ncks -v tp,cp {pjoin(path, name)} {pjoin(path, name_new)}')
            

            