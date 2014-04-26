import unittest
from scipy.io import loadmat
from numpy import array, asarray, squeeze
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
from kernel import kernel
from KernelRidgeRegression import KernelRidgeRegression

class KernelRidgeRegressionTestCase(unittest.TestCase):
    """Tests for `KernelRidgeRegression.py`."""
    def test_krr(self):

        # load data
        mat_file = loadmat('KRR_test.mat',squeeze_me=False)
        data = mat_file['x']
        obs = mat_file['y_n']

        k_name   = "gaussian"
        k_params = array( [0.5] )
        k = KernelType(k_name,k_params)

        krr1 = KernelRidgeRegression(k, -1)
        krr2 = KernelRidgeRegression(k, 0.2)
        krr1.process(data,obs)
        krr2.process(data,obs)

        est_obs1_t = krr1.predict(data)
        est_obs2_t = krr2.predict(data)

        # clean up
        est_obs1_t = squeeze(asarray(est_obs1_t))
        est_obs2_t = squeeze(asarray(est_obs2_t))

        # load data to compare
        mat_file = loadmat('test_krr',squeeze_me=True)
        est_obs1 = mat_file['est_obs1']
        est_obs2 = mat_file['est_obs2']

        # make sure these are all equal
        self.assertIsNone(assert_array_equal(est_obs1, est_obs1_t))
        self.assertIsNone(assert_array_equal(est_obs2, est_obs2_t))

if __name__ == '__main__':
    unittest.main()