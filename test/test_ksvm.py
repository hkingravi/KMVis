# unit testing for KernelSVM module
import unittest
from scipy.io import loadmat
from numpy import array, random, hstack
from numpy.testing import assert_array_equal
from numpy.linalg import norm

# import modules
import sys, os

# do imports 
from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.KernelSVM import KernelSVM

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

class KernelSVMTestCase(unittest.TestCase):
    """Tests for `KernelSVM.py`."""
    def test_kernel_svm(self):

        # set random seed to ensure reproducible pseudorandom behavior
        random.seed(3)
        tolerance = 1e-12 # accepted tolerance of difference         

        # load data for training
        ksvm_data_filename = 'kernel_trick_data.mat'
        ksvm_data_filepath = genloadstring(data_path,ksvm_data_filename)
        data_file = loadmat(ksvm_data_filepath,squeeze_me=False)

        ksvm_classes_filename = 'kernel_trick_classes.mat'
        ksvm_classes_filepath = genloadstring(data_path,ksvm_classes_filename)
        classes_file = loadmat(ksvm_classes_filepath,squeeze_me=False)

        data1 = data_file['x1']
        data2 = data_file['x2']
        labels1 = classes_file['c1']
        labels2 = classes_file['c2']

        # initialize kernel
        k_name   = "gaussian"
        k_params = array( [2] )
        k = KernelType(k_name,k_params)

        # initialize parameters
        C = 100
        tol = 0.001
        max_its = 100

        # initialize SVM object
        ksvm = KernelSVM(k,C,tol,max_its)

        # prepare, and pass in data
        data = hstack([data1,data2])
        labels = hstack([labels1,labels2])
        ksvm.process(data,labels)

        predicted_labels_t = ksvm.predict(data)
        xv_t, yv_t, vals_t = ksvm.evaluateboundary(data,100,100)

        # load data to compare
        ksvm_test_filename = 'test_ksvm.mat'
        ksvm_test_filepath = genloadstring(test_path,ksvm_test_filename)
        mat_file = loadmat(ksvm_test_filepath,squeeze_me=False)
        predicted_labels = mat_file['predicted_labels']
        xv = mat_file['xv']
        yv = mat_file['yv']
        vals = mat_file['vals']

        xv_diff = norm(xv - xv_t)
        yv_diff = norm(yv - yv_t)
        vals_diff = norm(vals - vals_t)

        # make sure these are all equal
        self.assertIsNone(assert_array_equal(predicted_labels, predicted_labels_t))
        self.assertTrue(xv_diff < tolerance)
        self.assertTrue(yv_diff < tolerance)
        self.assertTrue(vals_diff < tolerance)

if __name__ == '__main__':
    unittest.main()
