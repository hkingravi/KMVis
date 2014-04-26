# returns softmax cost associated to a given theta parameter 
# data assumed to be in column form 
from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs
from numpy import array
import numpy.random as rnd
from softmax_cost import *
def softmax_train(dim, nclasses, wdecay, data, labels):
    theta = 0.005 * (rnd.randn(nclasses*dim,1)).flatten(1)
    res = fmin_l_bfgs_b(softmax_cost, theta, fprime=softmax_grad, args=(nclasses, dim, wdecay, data, labels), approx_grad=False, bounds=None)
#    res = fmin_bfgs(softmax_cost, theta, fprime=softmax_grad, args=(nclasses, dim, wdecay, data, labels))
    model_theta = res[0] # get model parameters 

