# returns softmax cost associated to a given theta parameter
# data assumed to be in column form
from numpy import reshape, ones, squeeze, asarray, arange, dot, tile, multiply, array, transpose
from numpy import max as npmax, exp as npexp, sum as npsum, log as nplog
import numpy
from scipy.sparse import csr_matrix
from scipy.linalg import norm
# from scipy.optimize import minimize
from scipy.optimize import *
import numpy.random as rnd

import matplotlib.pyplot as plt


def softmax_cost( theta, nclasses, dim, wdecay, data, labels ):
    # unroll parameters from theta
    theta = reshape(theta,(dim, nclasses)) # This was wrong
    theta= theta.T
    nsamp = data.shape[1]

    # generate ground truth matrix
    onevals = squeeze(ones((1,nsamp)))
    rows = squeeze(labels)-1 # This was wrong
    cols = arange(nsamp)
    ground_truth = csr_matrix((onevals,(rows,cols))).todense()


    # compute hypothesis; use some in-place computations
    theta_dot_prod = dot(theta,data)
    theta_dot_prod = theta_dot_prod - numpy.amax(theta_dot_prod, axis=0) # This was wrong
    soft_theta = npexp(theta_dot_prod)
    soft_theta_sum = npsum(soft_theta,axis=0)
    soft_theta_sum = tile(soft_theta_sum,(nclasses,1))
    hyp = soft_theta/soft_theta_sum


    # compute cost
    log_hyp = nplog(hyp)
    temp = array(multiply(ground_truth,log_hyp))
    temp = npsum(npsum(temp,axis=1),axis=0)
    cost = (-1.0/nsamp)*temp + 0.5*wdecay*pow(norm(theta,'fro'),2)
    return cost

def softmax_grad( theta, nclasses, dim, wdecay, data, labels ):
    # unroll parameters from theta
    theta = reshape(theta,(dim, nclasses)) # Do this
    theta= theta.T
    nsamp = data.shape[1]

    # generate ground truth matrix
    onevals = squeeze(ones((1,nsamp)))
    rows = squeeze(labels)-1 # Here should -1 to align zero-indexing
    cols = arange(nsamp)

    ground_truth = csr_matrix((onevals,(rows,cols))).todense()
#     plt.imshow(ground_truth,interpolation='nearest')
#     plt.draw()
#     print ground_truth

    # compute hypothesis; use some in-place computations
    theta_dot_prod = dot(theta,data)
    theta_dot_prod = theta_dot_prod - numpy.amax(theta_dot_prod, axis=0) # This was wrong
    soft_theta = npexp(theta_dot_prod)
    soft_theta_sum = npsum(soft_theta,axis=0)
    soft_theta_sum = tile(soft_theta_sum,(nclasses,1))
    hyp = soft_theta/soft_theta_sum

    # compute gradient
    thetagrad = (-1.0/nsamp)*dot(ground_truth-hyp,transpose(data)) + wdecay*theta

    thetagrad = asarray(thetagrad)
    thetagrad = thetagrad.flatten(1)
    return thetagrad

def softmax_predict( theta, nclasses, dim,  data ):
    # unroll parameters from theta
    theta = reshape(theta,(dim, nclasses)) # Do this
    theta= theta.T

    # compute hypothesis; use some in-place computations
    theta_dot_prod = dot(theta,data)
    theta_dot_prod = theta_dot_prod - numpy.amax(theta_dot_prod, axis=0) # This was wrong
    soft_theta = npexp(theta_dot_prod)
    soft_theta_sum = npsum(soft_theta,axis=0)
    soft_theta_sum = tile(soft_theta_sum,(nclasses,1))
    hyp = soft_theta/soft_theta_sum
    print "hyp.shape"
    print hyp.shape
    pred=numpy.argmax(hyp, axis=0)

    return numpy.asarray(pred)

