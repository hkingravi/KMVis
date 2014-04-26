from numpy import array_str
import sys, os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../exceptions'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from Error import *
class KernelType:
    """
    Implements a class which initializes the kernel type and parameters.
    Currently supported kernels are

    * polynomial - :math:`k(x,y):= \\langle x,y \\rangle^{\gamma} + c`,
      where :math:`\ x,y,\in\mathbb{R}^d, \gamma \in\mathbb{N}, c\in\mathbb{R}`.
    * sigmoid - :math:`k(x,y):= \\tanh(\\alpha x^Ty + c)`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\alpha, c\in\mathbb{R}`.
    * gaussian - :math:`k(x,y):= \exp{\left(\\frac{\|x-y\|^2}{2\sigma^2}\\right)}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\sigma\in\mathbb{R}_+`.
    * laplacian - :math:`k(x,y):= \exp{\left(\\frac{\|x-y\|}{\sigma}\\right)}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\sigma\in\mathbb{R}_+`.
    * cauchy - :math:`k(x,y):= \\frac{1}{1 + \\frac{\|x-y\|^2}{\sigma^2}}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\sigma\in\mathbb{R}_+`.
    * periodic - :math:`k(x,y):= \exp{\left(\\frac{2\sin^2(\pi(x-y)/p)}{\sigma^2}\\right)}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, :math:`\\sigma\in\mathbb{R}_+` and :math:`p\in\mathbb{R}_+`.
    Example usage::

        k_name = "sigmoid"
        k_params = array( [0.5, 1.2] ) # numpy array
        k = KernelType(k_name,k_params)

    :param k_name: Kernel name
    :type k_name: string
    :param k_params: Kernel parameters
    :type k_params: numpy mat
    :return: `KernelType object`
    :rtype:  KernelType
    """
    def __init__(self, k_name, k_params):
        """
        Constructor for KernelType class.
        """
        # need to check if parameters passed are legitimate
        if k_name == "polynomial":
            k_size = k_params.shape
            if k_size[0] == 2:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: polynomial kernel needs degree and bias")
        elif k_name == "gaussian":
            k_size = k_params.shape
            if k_size[0] == 1:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: gaussian kernel needs bandwidth")
        elif k_name == "sigmoid":
            k_size = k_params.shape
            if k_size[0] == 2:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: sigmoid kernel needs alpha and bias")
        elif k_name == "laplacian":
            k_size = k_params.shape
            if k_size[0] == 1:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: laplacian kernel needs bandwidth")
        elif k_name == "cauchy":
            k_size = k_params.shape
            if k_size[0] == 1:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: cauchy kernel needs bandwidth")
        elif k_name == "periodic":
            k_size = k_params.shape
            if k_size[0] == 2:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: periodic kernel needs period and bandwidth")
        elif k_name == "locally_periodic":
            k_size = k_params.shape
            if k_size[0] == 2:
                self.name   = k_name
                self.params = k_params
            else:
                raise KernelParametersError("KernelParametersError: Incorrect number of parameters: locally periodic kernel needs period and bandwidth")
        else:
            raise KernelTypeError("KernelTypeError: Invalid kernel type: supported types are polynomial, gaussian, laplacian, cauchy, sigmoid, periodic, locally periodic")

    def __str__(self):
        """
        This function prints out the KernelType object.
        """
        return self.name + " kernel with parameters " + array_str(self.params)
