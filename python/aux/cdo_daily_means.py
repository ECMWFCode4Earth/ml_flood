import os

def cdo_daily_means(path, file_includes):
    """
    loops through the given directory and and executes "cdo dayavg *file_includes* file_out"
    """
    for name in os.listdir(path_to_data):
        if file_includes in name:
            file_out = f'{path_to_data}{name[:-3]}_day.nc'
            print(f'cdo dayavg {path_to_data}{name} {file_out}')
            os.system(f'cdo dayavg {path_to_data}{name} {file_out})

