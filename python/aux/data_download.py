import os, warnings, time
import cdsapi
import multiprocessing as mp

# do not forget to define your api key in the ~/.cdsapirc file

def build_request(kwargs, input_checking=True):
    """Check user's request for MARS mandatory fields
    to make valid CDS API retrievals.
    """
    kwargs = kwargs.copy()
    if kwargs['base_level'] == 'pressure' and 'pressure_level' not in kwargs:
        raise IOError('base_level is pressure, but pressure_level not in kwargs')
        
    
    mandatory_fields = ["product_type", "format", "variable", "year", "month"]
    if not input_checking:
        mandatory_fields = []
    
    assumed_args = {"day":    ["01", "02", "03", "04",
                               "05", "06", "07", "08",
                               "09", "10", "11", "12",
                               "13", "14", "15", "16",
                               "17", "18", "19", "20",
                               "21", "22", "23", "24",
                               "25", "26", "27", "28",
                               "29", "30", "31"],
                    "time":  ["00", "01", "02", "03", "04", "05",
                              "06", "07", "08", "09", "10", "11",
                              "12", "13","14", "15", "16", "17",
                              "18", "19", "20","21", "22", "23"]}
    assume_fields = assumed_args.keys()

    # input checks
    for key in mandatory_fields:  # add mandatory arguments
        if key not in kwargs: 
            raise ValueError(f'"{key}" not found in arguments, but is a mandatory field!')
    
    if kwargs['base_level'] == 'pressure' and 'pressure_level' not in kwargs:
        raise IOError('base_level is pressure, but pressure_level not in kwargs')
    
    request_name = f"reanalysis-era5-{kwargs.pop('base_level')}-levels"
    request = {}
    for key in mandatory_fields:
        #print(kwargs)
        request[key] = kwargs.pop(key)
            
    for key in list(kwargs):  # add optional arguments
        request[key] = kwargs.pop(key)
                                  
    for key in assume_fields:  # assume some arguments if not given
        if key not in request:
            warnings.warn(f'"{key}" not found in arguments, assuming {key}={assumed_args[key]}')
    return request
             
def list_of_str(a):
    l = a if isinstance(a, list) else list(a)
    return [str(a) for a in l]

def cds_optimized_retrieval(years, months,
			    dataset_name: str,
			    request_in: dict,
			    save_to_folder: str, 
                            N_parallel_requests=1):
    c = cdsapi.Client()
    if N_parallel_requests>1:
        p = mp.Pool(int(N_parallel_requests))
    
    # download era5 data with the cdsapi
    # data request efficiency is highest when executed on a monthly basis
    years = list_of_str(years) 
    months = list_of_str(months) 

    # loop over time range
    for y in sorted(years):
        for m in sorted(months):
            m = str(m).zfill(2)  # leading zero

            request = request_in.copy()
            request['year'] = y
            request['month'] = m
            
            request = build_request(request)         
            varnamefile = "".join(list(request["variable"])).replace(' ','')
            save_to_filename = f'{save_to_folder}/{dataset_name}_{varnamefile}_{y}_{m}.nc'

            # start a request for one month; only execute if file does not exist
            if not os.path.isfile(save_to_filename):
                p.apply_async(c.retrieve, args=(dataset_name, request, save_to_filename))
    p.close()
    p.join()
