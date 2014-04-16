# returns softmax cost associated to a given theta parameter 
# data assumed to be in column form 
from numpy import reshape, ones, squeeze, asarray, arange, dot, tile, multiply, array, transpose
from numpy import max as npmax, exp as npexp, sum as npsum, log as nplog
from scipy.sparse import csr_matrix
from scipy.linalg import norm
from scipy.optimize import minimize
import numpy.random as rnd
def softmax_cost( theta, nclasses, dim, wdecay, data, labels ):
    # unroll parameters from theta 
    theta = reshape(theta,(nclasses,dim))
    nsamp = data.shape[1]
    #print theta

    # generate ground truth matrix
    onevals = squeeze(ones((1,nsamp)))
    rows = squeeze(labels)
    cols = arange(nsamp)
    ground_truth = csr_matrix((onevals,(rows,cols))).todense()

    # compute hypothesis; use some in-place computations
    theta_dot_prod = dot(theta,data)
    theta_dot_prod = theta_dot_prod - npmax(theta_dot_prod)
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
    theta = reshape(theta,(nclasses,dim))
    nsamp = data.shape[1]
    #print theta

    # generate ground truth matrix
    onevals = squeeze(ones((1,nsamp)))
    rows = squeeze(labels)
    cols = arange(nsamp)
    ground_truth = csr_matrix((onevals,(rows,cols))).todense()

    # compute hypothesis; use some in-place computations
    theta_dot_prod = dot(theta,data)
    theta_dot_prod = theta_dot_prod - npmax(theta_dot_prod)
    soft_theta = npexp(theta_dot_prod)        
    soft_theta_sum = npsum(soft_theta,axis=0)
    soft_theta_sum = tile(soft_theta_sum,(nclasses,1))
    hyp = soft_theta/soft_theta_sum
       
    # compute gradient
    thetagrad = (-1.0/nsamp)*dot(ground_truth-hyp,transpose(data)) + wdecay*theta

    thetagrad = asarray(thetagrad)
    thetagrad = thetagrad.flatten(1)
    return thetagrad


