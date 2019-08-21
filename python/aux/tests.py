import unittest
import numpy as np
import xarray as xr
from utils_flowmodel import shift_and_aggregate


class TestMethods(unittest.TestCase):

    def test_shift_and_aggregate(self):
        """
        shift_and_aggregate
        """
        data = xr.DataArray(np.random.rand(10), dims=['time'],
                            coords=dict(time=('time', range(10))))
        shift =  2
        aggregate = 3
        a_all = shift_and_aggregate(data, shift, aggregate)
        
        t = 5
        t_shift = t-shift
        b = (data[t_shift]+data[t_shift-1]+data[t_shift-2])/3
        
        a = float(a_all[t])
        b = float(b)
        self.assertTrue(np.allclose(a, b, rtol=1e-05, atol=1e-08))

if __name__ == '__main__':
    unittest.main()