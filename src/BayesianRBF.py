from numpy import dot, eye, diag, ones, array, asarray, squeeze
#from numpy.random import multivariate_normal as mvnorm
import numpy.random as rnd
from scipy.linalg import lstsq, inv
from kernel import kernel
# import scipy
class BayesianRBF:
    """
    Implements a class constructing a (linear) 
    Bayesian Radial Basis Function network. Currently, only 
    models with a radial kernel and a specific bandwidth parameter 
    are supported. For us, this includes the Gaussian, Laplacian and
    Cauchy kernels. 
 
    Example usage::

        # load data using numpy
        mat_file = loadmat('BayesianRBF_test.mat',squeeze_me=False)
        data = mat_file['x']
        obs = mat_file['y_n']

        # initialize kernel
        k_name   = "gaussian"
        k_params = np.array( [0.5] )
        k = KernelType(k_name,k_params)

        brbf = BayesianRBF(k, 0.2)
        krr.process(data,obs)
        est_obs = krr.reduce(data)

    :param kernel: object specifying kernel and parameters
    :type kernel: KernelType
    :param noise: scalar :math:`\sigma\in\mathbb{R}_+`
    :type noise: float 
    :return: `BayesianRBF object`
    :rtype:  BayesianRBF
    """
    def __init__(self, k_instance, noise, centers):
        """
        This initialization function stores the kernel type, the value 
        of the noise parameter, and the centers for the network. 
        """       
        # first, you need to ensure that the correct kernel type is used
        if k_instance.name != "gaussian" and k_instance.name != "laplacian" and k_instance.name != "cauchy":
            raise Exception("ERROR: Incorrect kernel type. Can only use the Gaussian, Laplacian, or Cauchy kernels.")
        else: 
            DEFAULT_NOISE = 0 # set up default constants 
            self.k_type = k_instance # initialize kernel
            self.centers = centers # store centers
            self.centers_dim = centers.shape[0] # store dimensionality of centers
            self.ncent = centers.shape[1] # store number of centers

            # need to check if parameters passed are legitimate        
            if noise >= 0:
                # safe to initialize
                self.noise = noise
            else:
                print "noise must be greater than or equal to zero: using default of 0"
                self.noise = DEFAULT_NOISE

    def process(self, data, obs_vec):
        """
        Generate function network model. 

        :param phi_mat: Training data matrix :math:`\mathcal{X}\in\mathbb{R}^{d\\times n}`
        :type phi_mat: numpy array
        :param obs_vec: Observation vector :math:`y\in\mathbb{R}^{1 \\times n}`
        :type obs_vec: numpy array
        :return: none
        :rtype:  none
        """        
        # check consistency of data
        data_dim = data.shape[0]  
        obs_num = obs_vec.shape[1] 
        data_num = data.shape[1] 

        if data_dim != self.centers_dim:  
            raise Exception("Dimensions of data and centers must be the same")
        elif obs_num != data_num:            
            raise Exception("Number of samples for data and observations must be the same")
        else:
            # peel off parameters 
            ki = self.k_type
            bandwidth = ki.params[0]          

            # create kernel feature matrix 
            phi = kernel(data, self.centers, self.k_type)           
            self.alpha = 1/(pow(bandwidth,2))
            self.beta = 1/(pow(self.noise,2))

            # take into account both options
            if self.noise is 0:
                # S = inv(alpha*eye(m) + beta*(Phi'*Phi));
                # m_n = beta*(S*(Phi'*vals'));
                jitter = 0.00001 # add jitter factor to ensure inverse doesn't blow up 
            	self.precision = inv(jitter*eye(self.ncent) + self.beta*dot(phi.transpose(),phi))
            else:
            	self.precision = inv(self.alpha*eye(self.ncent) + self.beta*dot(phi.transpose(),phi))                

            self.mn = (dot(obs_vec,phi)).transpose()
            self.mn = self.beta*(dot(self.precision,self.mn))
            self.mn_aslist = array((self.mn).transpose())[0].tolist() # store as list for use in multivariate normal sampling
       
    def predict(self, te_data):
        """
        Use Bayesian RBF model to compute regression values at data points, as well as posterior variance. 

        :param D: Testing data matrix :math:`D\in\mathbb{R}^{m\\times d}`
        :type phi_mat: numpy array
        :return: `f`: estimated output values at data :math:`\hat{y}\in\mathbb{R}^{1 \\times m}`
        :rtype:  numpy array
        :return: `sigma`: estimated output values at data :math:`\hat{sigma}\in\mathbb{R}^{1 \\times m}`
        :rtype:  numpy array
        """ 
        k_test = kernel(te_data, self.centers, self.k_type)	
        dim_test = te_data.shape[0]
        ntest = te_data.shape[1]

        # make sure dimensionality is consistent 
        if dim_test != self.centers_dim:
            raise Exception("Dimensions of data and centers must be the same")
        else:
            # compute posterior mean
            f = (dot(k_test,self.mn)).transpose()
         
            # compute posterior variance 
            var_m = dot(k_test,self.precision)
            var_m = dot(var_m,k_test.transpose())
            var_f = (diag(var_m)).transpose()
            one_mat = ones((1,ntest))
            var_f = var_f + (1/self.beta)*one_mat

        return f, var_f

    def draw_rfunc(self, te_data):
        """
        Draw random function from current Bayesian RBF model. 

        :param D: Testing data matrix :math:`D\in\mathbb{R}^{d \\times m}`
        :type phi_mat: numpy array
        :return: `f`: estimated output values at data :math:`\hat{y}\in\mathbb{R}^{1 \\times m}`
        :rtype:  numpy array
        """ 
        # sample a random weight vector from the Bayesian model 
        w_random = rnd.multivariate_normal(self.mn_aslist,self.precision,1).transpose()
        k_test = kernel(self.centers, te_data, self.k_type)

        f_rand = dot(w_random.transpose(),k_test)
        f_rand = squeeze(asarray(f_rand))

        return f_rand
