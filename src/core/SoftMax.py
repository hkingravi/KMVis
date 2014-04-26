from numpy import reshape, ones, squeeze, arange, dot, tile, multiply, array, transpose
from numpy import max as npmax, exp as npexp, sum as npsum, log as nplog
from scipy.sparse import csr_matrix
from scipy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
from softmax_cost import *
import numpy.random as rnd
import numpy
class SoftMax:
    """
    Implements a class for computing the softmax classifier.

    Example usage::

        # load data
        mat_file = loadmat('mmd_data.mat',squeeze_me=False) # load data, and use matplotlib to plot it
        data1 = np.transpose(mat_file['data1']) # data from distribution 1
        data2 = np.transpose(mat_file['data2']) # data from distribution 1
        data3 = np.transpose(mat_file['data3']) # data from distribution 2

        # set up kernel
        k_name = "gaussian"
        k_params = array( [3] ) # numpy array
        k = KernelType(k_name, k_params)

        # initialize MeanMap
        mm_obj = MeanMap(k)
        mm_obj.process(data) # build map

        # compute maximum mean discrepancy between data1 and data2
        dist1 = mm_obj.mmd(data2)
        dist2 = mm_obj.mmd(data3)
        print "The MMD between data1 and data2 is " + str(dist1)
        print "The MMD between data1 and data3 is " + str(dist2)

    :param k_type: Kernel
    :type k_type: Kernel object
    :return: `MeanMap object`
    :rtype:  MeanMap
    """
    def __init__(self, inputdim, numclasses, wdecayparm):
        """
        Initialize default parameters.
        """
        self.dim = inputdim
        self.nclasses = numclasses
        self.wdecay = wdecayparm
        self.theta = 0.005 * (rnd.randn(self.nclasses*self.dim,1)).flatten(1)

    def cost(self, theta):
        """
        This function computes the cost associated to the softmax classifier.

        :param data: data matrix :math:`D_1\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of training samples
        :type data: numpy array
        """
        # unroll parameters from theta
        theta = reshape(theta,(self.dim,self.nclasses))
        theta=theta.T
        nsamp = self.data.shape[1]

        # generate ground truth matrix
        onevals = squeeze(ones((1,nsamp)))
        rows = squeeze(self.labels)-1
        cols = arange(nsamp)
        ground_truth = csr_matrix((onevals,(rows,cols))).todense()

        # compute hypothesis; use some in-place computations
        theta_dot_prod = dot(theta,self.data)
        theta_dot_prod = theta_dot_prod - numpy.amax(theta_dot_prod, axis=0)
        soft_theta = npexp(theta_dot_prod)
        soft_theta_sum = npsum(soft_theta,axis=0)
        soft_theta_sum = tile(soft_theta_sum,(self.nclasses,1))
        hyp = soft_theta/soft_theta_sum

        # compute cost
        log_hyp = nplog(hyp)
        temp = array(multiply(ground_truth,log_hyp))
        temp = npsum(npsum(temp,axis=1),axis=0)
        cost = (-1.0/nsamp)*temp + 0.5*self.wdecay*pow(norm(theta,'fro'),2)
        thetagrad = (-1.0/nsamp)*dot(ground_truth-hyp,transpose(self.data)) + self.wdecay*theta

        thetagrad = thetagrad.flatten(1)
        return cost, thetagrad

    def train(self,data,labels, maxIter):
        """
        This function trains the softmax classifier.

        :param data: data matrix :math:`D_1\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of training samples
        :type data: numpy array
        """
        xx, ff, dd = fmin_l_bfgs_b(softmax_cost, self.theta, fprime=softmax_grad, iprint=1,
              args=(self.nclasses, self.dim, self.wdecay, data, labels), approx_grad=0, bounds=None, maxfun=maxIter)

#          xx, ff, dd = fmin_l_bfgs_b(self.cost, self.theta, fprime=softmax_grad,
#               args=(self.nclasses, self.dim, self.wdecay, data, labels), approx_grad=0, bounds=None, maxfun=400)

        self.theta = reshape(xx,(self.dim,self.nclasses))
        self.theta = self.theta.T # store returned result
#         print "XX"
#         print xx
#
#         print "self.theta"
#         print self.theta
#
#         print "ff"
#         print ff
#
#         print "dd"
#         print dd

        return xx






