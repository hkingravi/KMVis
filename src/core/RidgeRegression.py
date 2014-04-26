from numpy import dot, mat, eye
from scipy.linalg import lstsq, inv 
class RidgeRegression:
    """
    Implements a class performing ridge regression, or Tikhonov regularization.
 
    Example usage::

        # load data using numpy
        mat_file = loadmat('OMP_test.mat',squeeze_me=False)
        data = mat_file['data']
        obs = mat_file['obs']

        rr = RidgeRegression(0) # instantiate object
        rr.process(data,obs) # learn regularized vector x

    :param lam: scalar :math:`\lambda\in\mathbb{R}_+`
    :type lam: float 
    :return: `RidgeRegression object`
    :rtype:  RidgeRegression
    """
    def __init__(self, lam):
        """
        This initialization function stores the value of lambda. 
        """
        # set up default constants 
        DEFAULT_LAMBDA = 0

        # need to check if parameters passed are legitimate        
        if lam >= 0:
            # safe to initialize
            self.lam = lam
        else:
            print "lam must be positive: using default of 0"
            self.lam = DEFAULT_LAMBDA

    def process(self, phi_mat, obs_vec):
        """
        Solve optimization problem to get :math:`\\alpha`. 

        :param phi_mat: Training data matrix :math:`\Phi\in\mathbb{R}^{n\\times d}`
        :type phi_mat: numpy array
        :param obs_vec: Observation vector :math:`y\in\mathbb{R}^{n}`
        :type obs_vec: numpy array
        :return: `alpha`: estimated state vector :math:\\alpha\in\mathbb{R}^{d}`
        :rtype:  numpy array
        """        
        # you have two options; if lam is zero, you can run least squares
        # otherwise, you have to invert the matrix
        phi_mat = mat(phi_mat)
        dim = phi_mat.shape[1]

        if self.lam is 0:
            self.alpha = lstsq(phi_mat, obs_vec)[0]
        else:
            temp_mat = dot(phi_mat.T, phi_mat) + self.lam*eye(dim)
            temp_mat2 = dot(phi_mat.T,obs_vec)
            self.alpha = dot(inv(temp_mat), temp_mat2)

        return self.alpha
        
    def reduce(self, data):
        """
        Use :math:`\\alpha` to compute values at data points. 

        :param D: Testing data matrix :math:`D\in\mathbb{R}^{m\\times d}`
        :type phi_mat: numpy array
        :return: `est_obs_vec`: estimated output values at data :math:`\hat{y}\in\mathbb{R}^{m}`
        :rtype:  numpy array
        """ 
        est_obs_vec = dot(data, self.alpha)
        return est_obs_vec



