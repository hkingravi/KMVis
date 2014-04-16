# quick script checking the numerical gradient for a given objective function

# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# matplotlib stuff
import matplotlib as mp
from matplotlib import rc
import matplotlib.pyplot as plt

# add class directory to path
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../data')

# our imports 
from KernelType import *
from computeNumericalGradient import *
from scipy.optimize import fmin_l_bfgs_b
from softmax_train import *
from softmax_cost import *
from SoftMax import *


data = np.array([[2, 3, 4],[3, 5, 6]])
labels = np.array([1, 1, 9])
#cost_vals = sm.cost(data, labels)

# load data for test 
mat_file = loadmat('../data/softmax_debug_test.mat',squeeze_me=False)
data = mat_file['inputData']
labels = mat_file['labels2']
theta = mat_file['theta']

print data.shape
print labels.shape

nclasses = 10
dim = data.shape[0]
wdecay = 0.0001

# need to flatten theta for input 
theta = theta.flatten(1)

print theta 

# compute cost 
cost = softmax_cost(theta, nclasses, dim, wdecay, data, labels)
thetagrad = softmax_grad(theta, nclasses, dim, wdecay, data, labels)

print cost
print thetagrad

# after this test is passed, we move to the actual softmax training
mat_file = loadmat('../data/mnist_images_softmax.mat',squeeze_me=False)
data = mat_file['inputData']
labels = mat_file['labels']

dim = data.shape[0]
sm = SoftMax(dim, nclasses, wdecay)
sm.train(data, labels)

#res = fmin_l_bfgs_b(softmax_cost, theta, fprime=softmax_grad, 
#              args=(nclasses, dim, wdecay, data, labels), approx_grad=False, bounds=None)


#softmax_train(dim, nclasses, wdecay, data, labels)

#print cost_vals[0]
#print cost_vals[1]



