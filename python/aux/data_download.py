import os, warnings, time
import cdsapi
import multiprocessing as mp

# do not forget to define your api key in the ~/.cdsapirc file

def list_of_str(a):
    l = a if isinstance(a, list) else list(a)
    return [str(a) for a in l]

class CDS_Dataset(object):
    def __init__(self, dataset_name: str, save_to_folder: str):
        """Define the dataset-name and where to save the files.

        Parameters
        ----------
            dataset_name : str
                the name of the dataset as requested by the CDS API.

            save_to_folder : str
                path to the folder where to save the downloaded files.
        """
        self.ds_name = dataset_name
        self.save_to_folder = save_to_folder
        self.api = cdsapi.Client()

    @staticmethod
    def build_request(kwargs, input_checking=True):
        """Check user's request for MARS mandatory fields
        to make valid CDS API retrievals.
        """
        kwargs = kwargs.copy()
        mandatory_fields = ["product_type", "variable", "format", "year", "month"]
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

        request = {}
        for key in mandatory_fields:
            #print(kwargs)
            request[key] = kwargs.pop(key)

        for key in list(kwargs):  # add optional arguments
            request[key] = kwargs.pop(key)

        for key in assume_fields:  # assume some arguments if not given
            if key not in request:
                warnings.warn(f'"{key}" not found in arguments, assuming {key}={assumed_args[key]}')
                request[key] = assumed_args[key]
        return request

    def get(self, years, months, request: dict, N_parallel_requests=1):
        """Retrieve data from CDS API.

        Parameters
        ----------

            years : str or list of str
                for example '2010'
            months : str or list of str
                for example '1' for January
            request_in : dict
                key, value pairs for the CDS API
        """
        if N_parallel_requests<1: N_parallel_requests=1
        p = mp.Pool(int(N_parallel_requests))

        # download era5 data with the cdsapi
        # data request efficiency is highest when executed on a monthly basis
        years = list_of_str(years)
        months = list_of_str(months)

        # loop over time range
        for y in sorted(years):
            for m in sorted(months):
                m = str(m).zfill(2)  # leading zero

                req = request.copy()
                req['year'] = y
                req['month'] = m

                req = self.build_request(req)
                varstr = ','.join(list(request['variable']))
                f_out = f'{self.save_to_folder}/{self.ds_name}_{varstr}_{y}_{m}.nc'

                # start a req for one month; only execute if file does not exist
                if not os.path.isfile(f_out):
                    p.apply_async(self.api.retrieve, args=(self.ds_name, req, f_out))
        p.close()
        p.join()
