# unit testing for kernel module
import unittest
from scipy.io import loadmat
from numpy import array, arange
from numpy.testing import assert_array_equal
# import modules
import sys, os
exception_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/exceptions'))
core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/core'))
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils'))
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))

if not exception_path in sys.path:
    sys.path.insert(1, exception_path)
if not core_path in sys.path:
    sys.path.insert(1, core_path)
if not test_path in sys.path:
    sys.path.insert(1, test_path)
if not utils_path in sys.path:
    sys.path.insert(1, utils_path)

from KernelType import KernelType
from kernel import kernel
from genloadstring import genloadstring

class KernelTestCase(unittest.TestCase):
    """Tests for `kernel.py`."""
    def test_kernel_numeric(self):
        k_name1   = "gaussian"
        k_name2   = "sigmoid"
        k_name3   = "polynomial"
        k_name4   = "laplacian"
        k_name5   = "cauchy"
        k_name6   = "periodic"
        k_name7   = "locally_periodic"
        k_params1 = array( [1.2] )
        k_params2 = array( [0.5, 1.2] )
        k_params3 = array( [2, 0] )
        k_params4 = array( [1.2] )
        k_params5 = array( [1.2] )
        k_params6 = array( [1.2, 0.5] )
        k_params7 = array( [0.5, 1] )

        k1 = KernelType(k_name1,k_params1)
        k2 = KernelType(k_name2,k_params2)
        k3 = KernelType(k_name3,k_params3)
        k4 = KernelType(k_name4,k_params4)
        k5 = KernelType(k_name5,k_params5)
        k6 = KernelType(k_name6,k_params6)
        k7 = KernelType(k_name7,k_params7)

        # compute kernels on data
        x = arange(-5,5,0.1)
        x_rad = arange(-3,7,0.1)
        y = array([2]) # y vals

        k_gauss_t = kernel(x_rad,y,k1)
        k_sigm_t = kernel(x,y,k2)
        k_poly_t = kernel(x,y,k3)
        k_lap_t = kernel(x_rad,y,k4)
        k_cauchy_t = kernel(x_rad,y,k5)
        k_periodic_t = kernel(x_rad,y,k6)
        k_locally_periodic_t = kernel(x_rad,y,k7)

        # load data to compare
        test_filename = 'test_kernel' # have to reuse the path name; some bug in the path on Ubuntu 14.04
        test_filepath = genloadstring(test_path,test_filename)
        mat_file = loadmat(test_filepath,squeeze_me=False)
        k_gauss = mat_file['k_gauss']
        k_sigm = mat_file['k_sigm']
        k_poly = mat_file['k_poly']
        k_lap = mat_file['k_lap']
        k_cauchy = mat_file['k_cauchy']
        k_periodic = mat_file['k_periodic']
        k_locally_periodic = mat_file['k_locally_periodic']

        # make sure these are all equal
        self.assertIsNone(assert_array_equal(k_gauss,k_gauss_t))
        self.assertIsNone(assert_array_equal(k_sigm,k_sigm_t))
        self.assertIsNone(assert_array_equal(k_poly,k_poly_t))
        self.assertIsNone(assert_array_equal(k_lap,k_lap_t))
        self.assertIsNone(assert_array_equal(k_cauchy,k_cauchy_t))
        self.assertIsNone(assert_array_equal(k_periodic,k_periodic_t))
        self.assertIsNone(assert_array_equal(k_locally_periodic,k_locally_periodic_t))

if __name__ == '__main__':
    unittest.main()
