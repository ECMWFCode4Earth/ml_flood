import os

def cdo_daily_means(path, file_includes):
    """
    loops through the given directory and and executes "cdo dayavg *file_includes* file_out"
    appends "dayavg" at the end of the filename
    Note: the total file number is currently hard coded to match our 37 years * 12 months lengths of the data
    """
    i = 0
    for name in os.listdir(path):
        if file_includes in name and 'dayavg' not in name:
            i += 1
            file_out = f'{path}{name[:-3]}_dayavg.nc'
            print(f'file {i} of {37*12} ...')
            os.system(f'cdo dayavg {path}{name} {file_out}')