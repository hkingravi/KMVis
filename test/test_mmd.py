import unittest
from scipy.io import loadmat
from numpy import array
from numpy.testing import assert_array_equal
# import modules
import sys, os
path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/exceptions'))
path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/core'))
path3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
path4 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

if not path1 in sys.path:
    sys.path.insert(1, path1)
if not path2 in sys.path:
    sys.path.insert(1, path2)
if not path3 in sys.path:
    sys.path.insert(1, path3)
if not path4 in sys.path:
    sys.path.insert(1, path4)

del path1
del path2
del path3
del path4

from KernelType import KernelType
from MeanMap import MeanMap

class MMDTestCase(unittest.TestCase):
    """Tests for `MMD.py`."""
    def test_mmd(self):
        # load data for training
        mat_file = loadmat('mmd_data.mat',squeeze_me=False)
        data1 = mat_file['data1'] # data from distribution 1
        data2 = mat_file['data2'] # data from distribution 1
        data3 = mat_file['data3'] # data from distribution 2

        # set up kernel
        k_name = "gaussian"
        k_params = array( [3] ) # numpy array
        k = KernelType(k_name, k_params)

        # initialize MeanMap
        mm_obj = MeanMap(k)
        mm_obj.process(data1) # build map

        # compute maximum mean discrepancy between data1 and data2
        dist1_t = mm_obj.mmd(data2)
        dist2_t = mm_obj.mmd(data3)

        # load data to compare
        mat_file = loadmat('test_mmd',squeeze_me=False)
        dist1 = mat_file['dist1']
        dist2 = mat_file['dist2']

        # make sure these are all equal
        self.assertIsNone(assert_array_equal(dist1, dist1_t))
        self.assertIsNone(assert_array_equal(dist2, dist2_t))

if __name__ == '__main__':
    unittest.main()