from numpy import shape, mat, kron, exp, zeros, ones, tanh, dot, power, hstack, power, sqrt
from scipy import isscalar
def simplesmo(data, classes, k_instance, C, tol, max_its):
    """
    An implementation of a simplified version of the SMO algorithm. This function
    is primarily called by the KernelSVM class. 

    Example usage::

        # create data
        data = array([[-1, -1, 1, 1],[-1, 1, -1, 1]])
        classes = array([-1,-1,1,1])

        # create kernel
        k_name   = "gaussian"
        k_params = array( [1.2] )
        k_obj = KernelType(k_name,k_params)

        # compute Lagrange multipliers and threshold 
        k_mat = kernel(data1,data2,k1)

    :param data: data matrix :math:`D\in\mathbb{R}^{d \\times n}`, where 
                     :math:`d` is the dimensionality and 
                     :math:`n` is the number of samples
    :type data: numpy mat
    :param classes: classes vector :math:`Y\in\mathbb{R}^{1 \\times n}`, where                      
                     :math:`n` is the number of samples
    :type classes: numpy mat
    :param k_instance: Kernel object
    :type k_instance: kernel
    :param C: positive cost parameter 
    :type C: float
    :param tol: tolerance
    :type tol: float
    :param max_its: positive integer indication maximum number of iterations 
    :type max_its: integer
    :return: alpha, vector of Lagrange multipliers :math:`\\alpha\in\mathbb{R}^{1 \\times n}`
    :rtype:  numpy mat
    :return: b, constant indicating bias :math:`b`
    :rtype:  float
    """
    # do basic type checking here

    # actual algorithm 
    nsamp = data.shape[1] # get number of samples
    alpha = zeros((1,nsamp))
    b = 0

    curr_it = 0

    while (curr_it < max_its):
      num_changed_alphas = 0
      for i in xrange(1,nsamp):
        # calculate Ei using equation (2)
        
      curr_it = curr_it + 1


    return alpha, b
