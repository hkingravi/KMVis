# unit testing for KPCA module
import unittest
#from random import random, seed
from scipy.io import loadmat
from numpy import array, asarray, squeeze, transpose, nonzero, hstack
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
from kernel import kernel
from KernelType import KernelType
from KPCA import KPCA

class KPCATestCase(unittest.TestCase):
    """Tests for `BayesianRBF.py`."""
    def test_kpca(self):
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
        mat_file = loadmat('separable_gmm.mat', squeeze_me=False)
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
        mat_file = loadmat('kernel_trick_data.mat', squeeze_me=False)
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
        mat_file = loadmat('test_kpca', squeeze_me=False)
        cauchy_proj = mat_file['cauchy_proj']
        gauss_proj = mat_file['gauss_proj']
        poly_proj = mat_file['poly_proj']

        # make sure these matrices are equal
        self.assertIsNone(assert_array_equal(cauchy_proj, cauchy_proj_t))
        self.assertIsNone(assert_array_equal(gauss_proj, gauss_proj_t))
        self.assertIsNone(assert_array_equal(poly_proj, poly_proj_t))


if __name__ == '__main__':
    unittest.main()