from numpy import dot, eye
from scipy.linalg import lstsq
from kernel import kernel
# import scipy
class KernelRidgeRegression:
    """
    Implements a class performing kernel ridge regression.
 
    Example usage::

        # load data using numpy
        mat_file = loadmat('KRR_test.mat',squeeze_me=False)
        data = mat_file['x']
        obs = mat_file['y_n']

        # initialize kernel
        k_name   = "gaussian"
        k_params = np.array( [0.5] )
        k = KernelType(k_name,k_params)


        krr = KernelRidgeRegression(k, 0.2)
        krr.process(data,obs)
        est_obs = krr.reduce(data)

    :param kernel: object specifying kernel and parameters
    :type kernel: KernelType
    :param lam: scalar :math:`\lambda\in\mathbb{R}_+`
    :type lam: float 
    :return: `KernelRidgeRegression object`
    :rtype:  KernelRidgeRegression
    """
    def __init__(self, k_instance, lam):
        """
        This initialization function stores the value of lambda. 
        """       
        DEFAULT_LAMBDA = 0 # set up default constants 
        self.k_type = k_instance # initialize kernel

        # need to check if parameters passed are legitimate        
        if lam >= 0:
            # safe to initialize
            self.lam = lam
        else:
            print "lam must be positive: using default of 0"
            self.lam = DEFAULT_LAMBDA

    def process(self, data, obs_vec):
        """
        Solve optimization problem to get :math:`\\alpha`. 

        :param phi_mat: Training data matrix :math:`\Phi\in\mathbb{R}^{d\\times n}`
        :type phi_mat: numpy array
        :param obs_vec: Observation vector :math:`y\in\mathbb{R}^{n}`
        :type obs_vec: numpy array
        :return: `alpha`: estimated state vector :math:\\alpha\in\mathbb{R}^{d}`
        :rtype:  numpy array
        """        
        # check consistency of data
        obs_num = obs_vec.shape[0] 
        data_num = data.shape[0] 
        
        if obs_num == data_num:            
            self.data = data

            # take into account both options
            if self.lam is 0:
                k_mat = kernel(data, data, self.k_type)                        
            else:
                dim = data.shape[1]
                k_mat = kernel(data, data, self.k_type) + self.lam*eye(dim)                    
         
            self.alpha = lstsq(k_mat, obs_vec.transpose())[0]
            return self.alpha
        else:
            print "ERROR: number of samples for data and observations must be the same"
       
    def reduce(self, te_data):
        """
        Use :math:`\\alpha` to compute values at data points. 

        :param D: Testing data matrix :math:`D\in\mathbb{R}^{m\\times d}`
        :type phi_mat: numpy array
        :return: `est_obs_vec`: estimated output values at data :math:`\hat{y}\in\mathbb{R}^{m}`
        :rtype:  numpy array
        """ 
        k_test = kernel(self.data, te_data, self.k_type)	
        coeff = dot((self.alpha).transpose(), k_test)
        return coeff



