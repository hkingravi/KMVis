from numpy import shape, mat, kron, exp, zeros, ones, tanh, dot, power, hstack, power
from scipy import isscalar
def kernel(data1, data2, k_type):
    """
    Compute the kernel function specified by the kernel type between two datasets 
    data1 and data2.

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
    :return: k_mat, kernel matrix :math:`K\in\mathbb{R}^{n \\times m}`
    :rtype:  numpy mat
    """
    k_name = k_type.name # need to check this type later
    k_params = k_type.params

    data1 = mat(data1)
    data2 = mat(data2)

    x_dims = data1.shape
    y_dims = data2.shape

    # subtlety: need to check if x and y are matrices or vectors
    # this is pretty tedious to check, but we need to do it because
    # of the retarded way numpy approaches this issue
    if len(x_dims) == 1:
        x_dims = list(x_dims)
        x_dims.insert(0, 1)
        x_dims = tuple(x_dims)
    if len(y_dims) == 1:
        y_dims = list(y_dims)
        y_dims.insert(0, 1)
        y_dims = tuple(y_dims)        

    k_mat = zeros((x_dims[1], y_dims[1]), float)  

    if x_dims[0] == y_dims[0]: 
        if k_name == "polynomial":
            degree = k_params[0] 
            bias = k_params[1]

            d_vals = mat(dot(data1.T, data2)) # compute dot product 
            d_vals = power(d_vals, degree)

            if isscalar(d_vals):
                bias_mat = mat(ones((d_vals.shape[0], d_vals.shape[1])))
                bias_mat = bias*bias_mat
                k_mat = d_vals + bias_mat
            else:                 
                bias_mat = mat(ones((d_vals.shape[0], d_vals.shape[1])))
                bias_mat = bias*bias_mat
                k_mat = d_vals + bias_mat
            
        elif k_name == "gaussian":
            sigma = k_params[0]

            d_vals = mat(dot(data1.T, data2))
            dx_vals = mat((power(data1, 2)).sum(axis=0))
            dy_vals = mat((power(data2, 2)).sum(axis=0))
            
            val1 = kron(ones((1, dy_vals.shape[1])), dx_vals.T)
            val2 = kron(ones((dx_vals.shape[1], 1)), dy_vals)

            k_mat = val1 + val2 - 2*d_vals

            s_val = -1.0/(2*pow(sigma, 2)) # compute scaling          
            k_mat = exp(s_val*k_mat)
        elif k_name == "sigmoid":
            alpha = k_params[0]
            bias = k_params[1]

            d_vals = alpha*mat(dot(data1.T, data2))
            c_vals = bias*ones((d_vals.shape[0], d_vals.shape[1]))           
 
            k_mat = tanh(d_vals + c_vals)
        else:
            print "Error: Invalid kernel type"
    else: 
        print "Error: data1 and data2 matrix dimensions do not match: see documentation"

    return k_mat
