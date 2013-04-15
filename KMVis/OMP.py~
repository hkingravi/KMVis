from numpy import shape, dot, vdot, absolute, argmax, zeros
from scipy.linalg import norm, lstsq
from scipy import isscalar
# import scipy
class OMP:
    """
    Implements a class performing Orthogonal Matching Pursuit. OMP is just one algorithm 
    for basis pursuit, which attempts to find a solution to the sparse recovery problem 

    .. math::

       \\begin{align}
       \min \|x\|_1 \\text{ s.t. } \Phi x = y, 
       \\end{align}

    where :math:`\Phi\in\mathbb{R}^{m\\times n}` is the dictionary, :math:`y\in\mathbb{R}^{m}`
    is the vector of observations, and :math:`x\in\mathbb{R}^{n}` is the 'state' vector, which
    is assumed to be sparse. Typically, :math:`m \ll n`, i.e. the number of observations are much
    smaller than the elements in the dictionary; due to this property, the matrix is underdetermined,
    and the sparsity requirement functions as a prior that enables the state's recovery. 

    For a theoretical analysis and outline of the algorithm, see 
    `Signal Recovery From Random Measurements Via Orthogonal Matching Pursuit. 
    <http://authors.library.caltech.edu/9490/1/TROieeetit07.pdf>`_ by Tropp and Gilbert.
 
    Example usage::

        # load data using numpy
        mat_file = loadmat('OMP_test.mat',squeeze_me=False)
        data = mat_file['data']
        obs = mat_file['obs']

        omp = OMP(1e-7,10000) # instantiate object
        omp.process(data,obs) # learn sparse vector x

    :param tol: Residual error tolerance
    :type tol: float
    :param max_iter: Maximum number of iterations to run OMP
    :type max_iter: int
    :return: `OMP object`
    :rtype:  OMP
    """
    def __init__(self, tol, max_iter):
        """
        This initialization function stores the residual tolerance and the maximum
        number of iterations allowed. 
        """
        # set up default constants 
        DEFAULT_TOL = 1e-7
        DEFAULT_MAX_ITER = 10000

        # need to check if parameters passed are legitimate        
        if tol > 0 and max_iter > 0:
            # safe to initialize
            self.tol = tol
            self.max_iter = max_iter
        elif tol > 0:
            print "Max iterations must be positive: \
                   using default of 10,000"
            self.tol = tol 
            self.max_iter = DEFAULT_MAX_ITER
        elif max_iter > 0:
            print "Tolerance must be positive: \
                   using default of 1e-7"
            self.tol = DEFAULT_TOL 
            self.max_iter = max_iter
        else:
            print "Max iterations and tolerance must be positive: using defaults of \
                   (tol = 1e-7, max_iter = 10,000)"
            self.tol = DEFAULT_TOL          
            self.max_iter = DEFAULT_MAX_ITER     

    def process(self, phi_mat, obs_vec):
        """
        Solve optimization problem. 

        :param phi_mat: Dictionary :math:`\Phi\in\mathbb{R}^{m\\times n}`
        :type phi_mat: numpy array
        :param obs_vec: Observation vector :math:`y\in\mathbb{R}^{m}`
        :type obs_vec: numpy array
        :return: `x_hat`: estimated state vector :math:`x\in\mathbb{R}^{n}`
        :rtype:  numpy array
        :return: `current_it`: number of iterations algorithm is run
        :rtype:  int     
        """        
        nsamp = phi_mat.shape[1]
        resid = obs_vec
        indices = [] 
        current_it = 0
        norm_r = norm(resid)

        while norm_r > self.tol and current_it < self.max_iter:
            proxy = dot(phi_mat.transpose(), resid) # compute proxy
            gamma = argmax(absolute(proxy)) # find index with largest value
            indices.append(gamma) # update indices

            phi_curr = phi_mat[:, indices].copy() # select columns from phi_mat 

            x_hat_sub = zeros((len(indices), 1))
            x_hat = zeros((nsamp, 1))

            if len(indices) is 1: # in this case, need to solve least squares yourself
                x_hat_sub = vdot(phi_curr, phi_curr)*vdot(phi_curr, obs_vec)
            else:            
                x_hat_sub = lstsq(phi_curr, obs_vec)[0]
                
            index = 0
   
            for i in indices:   
                if isscalar(x_hat_sub):
                    x_hat[i] = x_hat_sub
                else: 
                    x_hat[i] = x_hat_sub[index]
                index = index + 1

            resid = obs_vec - dot(phi_mat, x_hat) # calculate new residual 
            norm_r = norm(resid)
            current_it = current_it + 1 

            self.x_hat = x_hat
            
        return x_hat, current_it

        
