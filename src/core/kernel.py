from numpy import multiply, array
from kernel_base import kernel_base
from KernelType import KernelType
def kernel(data1, data2, k_type):
    """
    Compute kernel functions specified by the user between two datasets
    data1 and data2. This function utilizes the kernel_base function, and
    has the capacity to create complex kernels composed of other kernel types.

    Example usage::

        # create data
        data1 = array([[1, 2, 3],[3, 4, 5]])
        data2 = array([[2, 1],[1, 0]])

        # create kernel
        k_name   = "gaussian"
        k_params = array( [1.2] )
        k_obj = KernelType(k_name,k_params)

        # generate kernel matrix
        k_mat = kernel(data1,data2,k1)

    :param data1: first data matrix :math:`D_1\in\mathbb{R}^{d \\times m}`, where
                     :math:`d` is the dimensionality and
                     :math:`m` is the number of samples
    :type data1: numpy mat
    :param data2: first data matrix :math:`D_2\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of samples
    :type data2: numpy mat
    :param k_type: Kernel object
    :type k_type: kernel
    :return: k_mat, kernel matrix :math:`K\in\mathbb{R}^{m \\times n}`
    :rtype:  numpy mat
    """
    k_name = k_type.name # need to check this type later
    k_params = k_type.params

    # if base kernels, simply call it
    if (k_name == "polynomial" or k_name == "gaussian" or
          k_name == "laplacian" or k_name == "cauchy" or
          k_name == "sigmoid" or k_name == "periodic"):
        k_mat = kernel_base(data1, data2, k_type)
    elif k_name == "locally_periodic":
        k_name1 = "gaussian"
        k_name2 = "periodic"
        k_params1 = array([k_params[0]])
        k_params2 = k_params
        k_type_gauss = KernelType(k_name1,k_params1)
        k_type_periodic = KernelType(k_name2,k_params2)
        k_mat = multiply(kernel_base(data1, data2, k_type_periodic),kernel_base(data1, data2, k_type_gauss))
    else:
        raise Exception("Invalid kernel type")
    return k_mat

