# unit testing for BayesianRBF module
import unittest
#from random import random, seed
from scipy.io import loadmat
from numpy.linalg import svd
from numpy import array, dot, diag, asarray, zeros, random
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

# our imports
from BayesianRBF import BayesianRBF
from kernel import kernel
from KernelType import KernelType

class BayesianRBFTestCase(unittest.TestCase):
    """Tests for `BayesianRBF.py`."""
    def test_brbf(self):
        random.seed(3)
        mat_file = loadmat('BRBF_test.mat',squeeze_me=False)
        data = mat_file['data']
        obs = mat_file['cvals']
        centers = mat_file['centers']
        eval_data = mat_file['eval_data']
        evals = mat_file['evals']

        k_name   = "gaussian"
        k_params = array( [0.5] )
        k = KernelType(k_name,k_params)

        brbf1 = BayesianRBF(k, 0.2, centers)
        brbf1.process(data,obs)

        f, var_f = brbf1.predict(data)

        # plot random functions
        num_rfunc = 40
        ntest = eval_data.shape[1]
        A = zeros((num_rfunc,ntest))

        for i in xrange(0, num_rfunc):
            # draw random function
            f_rand = brbf1.draw_rfunc(eval_data)

            # to save f_rand, you need to reshape appropriately
            f_rand = asarray(f_rand)
            f_rand = f_rand.reshape(1,ntest)
            A[i,:] = f_rand

        # now, generate basis for random functions using SVD
        rank1 = 1
        rank2 = 3
        rank3 = 8
        U, S, V = svd(A, full_matrices=True)

        # create low-rank matrices
        S = diag(S)
        U_ret1 = U[:,0:rank1]
        V_ret1 = V[0:rank1,:]
        S_ret1 = S[0:rank1,0:rank1]

        U_ret2 = U[:,0:rank2]
        V_ret2 = V[0:rank2,:]
        S_ret2 = S[0:rank2,0:rank2]

        U_ret3 = U[:,0:rank3]
        V_ret3 = V[0:rank3,:]
        S_ret3 = S[0:rank3,0:rank3]

        # reconstruct
        A_r1_t = dot(U_ret1,dot(S_ret1,V_ret1))
        A_r2_t = dot(U_ret2,dot(S_ret2,V_ret2))
        A_r3_t = dot(U_ret3,dot(S_ret3,V_ret3))

        # check against stored files
        mat_file = loadmat('test_brbf',squeeze_me=False)
        A_r1 = mat_file['A_r1']
        A_r2 = mat_file['A_r2']
        A_r3 = mat_file['A_r3']

        # make sure these matrices are equal
        self.assertIsNone(assert_array_equal(A_r1,A_r1_t))
        self.assertIsNone(assert_array_equal(A_r2,A_r2_t))
        self.assertIsNone(assert_array_equal(A_r3,A_r3_t))

if __name__ == '__main__':
    unittest.main()