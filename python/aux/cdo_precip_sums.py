import os

def cdo_precip_sums(path, file_includes='precipitation'):
    """
    loops through the given directory and and executes "cdo -b 32 daysum filein.nc fileout.nc"
    appends "daysum" at the end of the filename
    Note: the total file number is currently hard coded to match our 37 years * 12 months lengths of the data
    """
    i = 0
    for name in os.listdir(path):
        if file_includes in name and 'day' not in name and 'daysum' not in name:
            i += 1
            file_out = f'{path}{name[:-3]}_daysum.nc'
            print(f'file {i} of {37*12} ...')
            os.system(f'cdo -b 32 daysum {path}{name} {file_out}')