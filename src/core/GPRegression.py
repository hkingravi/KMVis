from numpy import dot, eye, diag, ones, array, asarray, squeeze
#from numpy.random import multivariate_normal as mvnorm
import numpy.random as rnd
from scipy.linalg import lstsq, cholesky
from kernel import kernel

class GPRegression:
    """
    A class implementation of Gaussian process regression. Currently, only
    models with a radial kernel and a specific bandwidth parameter
    are supported. Currently, only a Gaussian likelihood is considered,
    although future implementations will support non-Gaussian likelihoods
    for robustness purposes.

    Example usage::

        # load data using numpy
        mat_file = loadmat('gpregression_test.mat',squeeze_me=False)
        data = mat_file['x']
        obs = mat_file['y_n']

        # initialize kernel
        k_name   = "gaussian"
        k_params = np.array( [0.5] )
        k = KernelType(k_name,k_params)
        noise = 0.2

        gpr = GPRegression(k, noise)
        gpr.process(data,obs)
        est_obs = gpr.predict(data)

    :param kernel: object specifying kernel and parameters
    :type kernel: KernelType
    :param noise: scalar :math:`\sigma\in\mathbb{R}_+`
    :type noise: float
    :return: `BayesianRBF object`
    :rtype:  BayesianRBF
    """
    def __init__(self, k_instance, noise):
        """
        This initialization function stores the kernel type, the value
        of the noise parameter, and the centers for the network.
        """
        # first, you need to ensure that the kernel is radially symmetric
        if (k_instance.name != "gaussian" and k_instance.name != "laplacian" and
            k_instance.name != "cauchy" and k_instance.name != "periodic" and
            k_instance.name != "locally_periodic"):
            raise Exception("ERROR: Incorrect kernel type. Can only use "
                            "the Gaussian, Laplacian, Cauchy, periodic or locally kernels.")
        else:
            DEFAULT_NOISE = 0 # set up default constants
            self.k_type = k_instance # initialize kernel

            # need to check if parameters passed are legitimate
            if noise >= 0:
                # safe to initialize
                self.noise = noise
            else:
                print "noise must be greater than or equal to zero: using default of 0"
                self.noise = DEFAULT_NOISE

            # these attributes are initialized later
            self.lmat = None
            self.mean_vec = None
            self.data = None

    def process(self, data_in, obs_vec):
        """
        Generate function network model.

        :param data: Training data matrix :math:`\mathcal{X}\in\mathbb{R}^{d\\times n}`
        :type data: numpy array
        :param obs_vec: Observation vector :math:`y\in\mathbb{R}^{1 \\times n}`
        :type obs_vec: numpy array
        :return: none
        :rtype:  none
        """
        # check consistency of data
        obs_num = obs_vec.shape[1]
        data_num = data_in.shape[1]

        if obs_num != data_num:
            raise Exception("Number of samples for data and observations must be the same")
        else:
            # initialize variables
            self.data = data_in
            self.data_dim = data_in.shape[0]
            nsamp = data_num

            # peel off parameters
            ki = self.k_type
            bandwidth = ki.params[0]

            # compute regularized kernel matrix
            kmat = kernel(self.data, self.data, self.k_type) + (pow(self.noise,2))*eye(nsamp)

            # perform Cholesky factorization, and compute mean vector (for stable inverse computations)
            self.lmat = cholesky(kmat).transpose()
            self.mean_vec = lstsq(self.lmat, obs_vec)
            self.mean_vec = lstsq(self.lmat.transpose(), self.mean_vec)

    def predict(self, te_data):
        """
        Use GP model to compute regression values at data points, as well as posterior variance.

        :param D: Testing data matrix :math:`D\in\mathbb{R}^{m\\times d}`
        :type phi_mat: numpy array
        :return: `f`: estimated output values at data :math:`\hat{y}\in\mathbb{R}^{1 \\times m}`
        :rtype:  numpy array
        :return: `sigma`: estimated output values at data :math:`\hat{sigma}\in\mathbb{R}^{1 \\times m}`
        :rtype:  numpy array
        """
        k_test = kernel(self.data, te_data, self.k_type)
        dim_test = te_data.shape[0]
        ntest = te_data.shape[1]

        # make sure dimensionality is consistent
        if dim_test != self.data_dim:
            raise Exception("Dimensions of training and test data must be the same")
        else:
            # compute posterior mean
            f = dot(k_test,self.mean_vec)

            # compute posterior variance
            var_m = lstsq(self.lmat, k_test)
            var_f = kernel(te_data,te_data,self.k_type) - dot(var_m,var_m)

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
