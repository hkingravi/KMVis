# unit testing for KernelSVM module
import unittest
from scipy.io import loadmat
from numpy import array, random, hstack
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
from KernelSVM import KernelSVM

class KernelSVMTestCase(unittest.TestCase):
    """Tests for `KernelSVM.py`."""
    def test_kernel_svm(self):

        # set random seed to ensure reproducible pseudorandom behavior
        random.seed(3)

        # load data for training
        data_file = loadmat('kernel_trick_data.mat',squeeze_me=False)
        classes_file = loadmat('kernel_trick_classes.mat',squeeze_me=False)

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
        mat_file = loadmat('test_ksvm',squeeze_me=False)
        predicted_labels = mat_file['predicted_labels']
        xv = mat_file['xv']
        yv = mat_file['yv']
        vals = mat_file['vals']

        # make sure these are all equal
        self.assertIsNone(assert_array_equal(predicted_labels, predicted_labels_t))
        self.assertIsNone(assert_array_equal(xv, xv_t))
        self.assertIsNone(assert_array_equal(yv, yv_t))
        self.assertIsNone(assert_array_equal(vals, vals_t))

if __name__ == '__main__':
    unittest.main()