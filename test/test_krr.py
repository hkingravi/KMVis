import unittest
from scipy.io import loadmat
from numpy import array, asarray, squeeze
from numpy.linalg import norm

# import modules
import sys, os

# do imports 
from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.KernelRidgeRegression import KernelRidgeRegression

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

class KernelRidgeRegressionTestCase(unittest.TestCase):
    """Tests for `KernelRidgeRegression.py`."""
    def test_krr(self):
        tolerance = 1e-12 # accepted tolerance of difference 

        # load data
        krr_data_filename = 'KRR_test.mat'
        krr_data_filepath = genloadstring(data_path,krr_data_filename)
        mat_file = loadmat(krr_data_filepath,squeeze_me=False)
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
        krr_test_filename = 'test_krr.mat'
        krr_test_filepath = genloadstring(test_path,krr_test_filename)
        mat_file = loadmat(krr_test_filepath,squeeze_me=True)
        est_obs1 = mat_file['est_obs1']
        est_obs2 = mat_file['est_obs2']

        est_diff1 = norm(est_obs1 - est_obs1_t)
        est_diff2 = norm(est_obs2 - est_obs2_t)
 
        print est_diff1
        print est_diff2

        # make sure these are all equal
        self.assertTrue(est_diff1 < tolerance)
        self.assertTrue(est_diff2 < tolerance)

if __name__ == '__main__':
    unittest.main()
