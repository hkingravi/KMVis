# unit testing for BayesianRBF module
import unittest
#from random import random, seed
from scipy.io import loadmat
from numpy.linalg import svd
from numpy import array, dot, diag, asarray, zeros, random
from numpy.linalg import norm

# import modules
import sys, os

# do imports 
from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.BayesianRBF import BayesianRBF

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

class BayesianRBFTestCase(unittest.TestCase):
    """Tests for `BayesianRBF.py`."""
    def test_brbf(self):
        tolerance = 1e-12 # accepted tolerance of difference 

        random.seed(3)
        brbf_data_filename = 'BRBF_test.mat'
        brbf_data_filepath = genloadstring(data_path,brbf_data_filename)
        brbf_mat_file = loadmat(brbf_data_filepath,squeeze_me=False)
        data = brbf_mat_file['data']
        obs = brbf_mat_file['cvals']
        centers = brbf_mat_file['centers']
        eval_data = brbf_mat_file['eval_data']
        evals = brbf_mat_file['evals']

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
        brbf_test_filename = 'test_brbf.mat'
        brbf_test_filepath = genloadstring(test_path,brbf_test_filename)
        brbf_test_mat_file = loadmat(brbf_test_filepath,squeeze_me=False)
        A_r1 = brbf_test_mat_file['A_r1']
        A_r2 = brbf_test_mat_file['A_r2']
        A_r3 = brbf_test_mat_file['A_r3']

        A_r1_diff = norm(A_r1-A_r1_t)
        A_r2_diff = norm(A_r2-A_r2_t)
        A_r3_diff = norm(A_r3-A_r3_t)

        # make sure these matrices are equal
        self.assertTrue(A_r1_diff < tolerance)
        self.assertTrue(A_r2_diff < tolerance)
        self.assertTrue(A_r3_diff < tolerance)

if __name__ == '__main__':
    unittest.main()
