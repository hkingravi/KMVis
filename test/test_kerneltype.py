# unit testing for KernelType module
import unittest
from numpy import array

# import modules
import sys, os
path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/exceptions'))
path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/core'))
if not path1 in sys.path:
    sys.path.insert(1, path1)
if not path2 in sys.path:
    sys.path.insert(1, path2)
del path1
del path2

from Error import KernelParametersError
from Error import KernelTypeError
from KernelType import KernelType

class KernelTypeTestCase(unittest.TestCase):
    """Tests for `KernelType.py`."""

    def test_wrong_dim_initialization(self):
        """Check to see if kernels detect wrong dimensionality"""
        k_gauss = "gaussian"
        k_sig   = "sigmoid"
        k_poly  = "polynomial"
        k_lap   = "laplacian"
        k_cau   = "cauchy"
        k_per   = "periodic"
        k_lper  = "locally_periodic"

        parms1 = array( [1] )
        parms2 = array( [1, 2] )

        # only doing this because assertEqual doesn't seem to work
        try:
            KernelType(k_gauss,parms2)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: gaussian kernel needs bandwidth", e.message)
        try:
            KernelType(k_lap,parms2)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: laplacian kernel needs bandwidth", e.message)
        try:
            KernelType(k_cau,parms2)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: cauchy kernel needs bandwidth", e.message)
        try:
            KernelType(k_sig,parms1)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: sigmoid kernel needs alpha and bias", e.message)
        try:
            KernelType(k_poly,parms1)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: polynomial kernel needs degree and bias", e.message)
        try:
            KernelType(k_per,parms1)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: periodic kernel needs period and bandwidth", e.message)
        try:
            KernelType(k_lper,parms1)
        except KernelParametersError, e:
            self.assertEqual("KernelParametersError: Incorrect number of parameters: locally periodic kernel needs period and bandwidth", e.message)

    def test_wrong_kernel_initialization(self):
        """Check to see if kernels detect wrong kernel type"""
        k_wrong   = "wrong_name"
        parms = array( [1] )

        try:
            KernelType(k_wrong,parms)
        except KernelTypeError, e:
            self.assertEqual("KernelTypeError: Invalid kernel type: supported types are polynomial, gaussian, laplacian, cauchy, sigmoid, periodic, locally periodic", e.message)

if __name__ == '__main__':
    unittest.main()