# unit testing for KPCA module
import unittest
# from random import random, seed
from scipy.io import loadmat
from numpy import array, asarray, squeeze, transpose, nonzero, hstack
from numpy.linalg import norm

# import modules
import sys, os

# do imports 
from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.KPCA import KPCA

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

class KPCATestCase(unittest.TestCase):
    """Tests for `BayesianRBF.py`."""
    def test_kpca(self):
        tolerance = 1e-12 # accepted tolerance of difference 

        k_name1   = "gaussian"
        k_name2   = "cauchy"
        k_name3   = "polynomial"

        k_params1 = array( [10] )
        k_params2 = array( [5] )
        k_params3 = array( [2, 0] )

        k_gauss = KernelType(k_name1, k_params1)
        k_cauchy = KernelType(k_name2, k_params2)
        k_poly = KernelType(k_name3, k_params3)

        # load GMM data
        gmm_data_filename = 'separable_gmm'
        gmm_data_filepath = genloadstring(data_path,gmm_data_filename)
        mat_file = loadmat(gmm_data_filepath, squeeze_me=False)
        data = transpose(mat_file['data'])
        labels = mat_file['labels']

        # get and clean labels of data
        l1 = asarray(nonzero(labels == 1))
        l2 = asarray(nonzero(labels == 2))
        l3 = asarray(nonzero(labels == 3))
        l4 = asarray(nonzero(labels == 4))

        l1 = l1[0, :]
        l2 = l2[0, :]
        l3 = l3[0, :]
        l4 = l4[0, :]

        # now load polynomial kernel data
        kt_data_filename = 'kernel_trick_data'
        kt_data_filepath = genloadstring(data_path,kt_data_filename)
        mat_file = loadmat(kt_data_filepath, squeeze_me=False)
        polydata1 = mat_file['x1']
        polydata2 = mat_file['x2']
        polydata = hstack([polydata1, polydata2])


        # compute KPCA on GMM data using cauchy kernel
        neigs = 3
        centered = 0

        kpca_cauchy = KPCA(k_cauchy, neigs, centered)
        kpca_cauchy.process(data)
        cauchy_proj_t = kpca_cauchy.project(data)

        # compute KPCA on GMM data using gaussian kernel
        kpca_gauss = KPCA(k_gauss, neigs, centered)
        kpca_gauss.process(data)
        gauss_proj_t = kpca_gauss.project(data)

        # compute KPCA on nonseparable data using polynomial kernel
        kpca_poly = KPCA(k_poly, neigs, centered)
        kpca_poly.process(polydata)
        poly_proj1 = kpca_poly.project(polydata1)
        poly_proj2 = kpca_poly.project(polydata2)

        polyproj1 = squeeze(asarray(poly_proj1))
        polyproj2 = squeeze(asarray(poly_proj2))
        poly_proj_t = hstack([polyproj1, polyproj2])

        # now load solution to compare against
        kpca_test_filename = 'test_kpca'
        kpca_test_filepath = genloadstring(test_path,kpca_test_filename)
        mat_file = loadmat(kpca_test_filepath, squeeze_me=False)
        cauchy_proj = mat_file['cauchy_proj']
        gauss_proj = mat_file['gauss_proj']
        poly_proj = mat_file['poly_proj']

        # compute norm difference 
        cauchy_diff = norm(cauchy_proj - cauchy_proj_t)
        gauss_diff = norm(gauss_proj - gauss_proj_t)
        poly_diff = norm(poly_proj - poly_proj_t)

        # make sure these matrices are equal
        self.assertTrue(cauchy_diff < tolerance)
        self.assertTrue(gauss_diff < tolerance)
        self.assertTrue(poly_diff < tolerance)


if __name__ == '__main__':
    unittest.main()
