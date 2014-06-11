import unittest
from scipy.io import loadmat
from numpy import array
from numpy.testing import assert_array_equal
# import modules
import sys, os

# do imports 
from ..src.core.KernelType import KernelType
from ..src.core.MeanMap import MeanMap

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

class MMDTestCase(unittest.TestCase):
    """Tests for `MMD.py`."""
    def test_mmd(self):
        # load data for training
        mmd_data_filename = 'mmd_data.mat'
        mmd_data_filepath = genloadstring(data_path,mmd_data_filename)
        mmd_mat_file = loadmat(mmd_data_filepath,squeeze_me=False)
        data1 = mmd_mat_file['data1'] # data from distribution 1
        data2 = mmd_mat_file['data2'] # data from distribution 1
        data3 = mmd_mat_file['data3'] # data from distribution 2

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
        mmd_test_filename = 'test_mmd.mat'
        mmd_test_filepath = genloadstring(test_path,mmd_test_filename)
        mmd_test_mat_file = loadmat(mmd_test_filepath,squeeze_me=False)
        dist1 = mmd_test_mat_file['dist1']
        dist2 = mmd_test_mat_file['dist2']

        # make sure these are all equal
        self.assertIsNone(assert_array_equal(dist1, dist1_t))
        self.assertIsNone(assert_array_equal(dist2, dist2_t))

if __name__ == '__main__':
    unittest.main()
