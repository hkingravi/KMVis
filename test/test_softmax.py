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
# from test.softmax_cost import softmax_grad, softmax_cost
sys.path.insert(0, '../src')
sys.path.insert(0, '../data')

# our imports
from KernelType import *
# from computeNumericalGradient import *
from scipy.optimize import fmin_l_bfgs_b
from softmax_train import *
from softmax_cost import *
from SoftMax import *
import time

#==============================================================================
## [Test softmax cost]

# load data for test
mat_file = loadmat('../softmax_debug_test.mat',squeeze_me=False)
data = mat_file['inputData']
labels = mat_file['labels']
theta=mat_file['theta']
lambda_=mat_file['lambda']


# -----------------------------------------------------------------------------
## Unit test of softmax_cost and softmax_grad, compared to the MATLABT version
cost=softmax_cost(theta, 10, 8, lambda_, data, labels)
grad=softmax_grad(theta, 10, 8, lambda_, data, labels)
thetagrad = reshape(grad,(8,10)) # Do this
thetagrad= thetagrad.T
plt.plot(thetagrad)
plt.draw()



#===============================================================================
## [Release Test -- training]

# after this test is passed, we move to the actual softmax training
print "[mnist images example]"
mat_file = loadmat('../softmax_release_test.mat',squeeze_me=False)
data = mat_file['inputData']
labels = mat_file['labels']
theta=mat_file['theta']
wdecay=mat_file['lambda']
nclasses = 10
dim = data.shape[0]

print "=========================\n input Parms"
print "inputData"
print data.shape
print "labels: "
print labels.shape
print "dim:"
print dim
print "nclasses:"
print nclasses
print "========================="


sm = SoftMax(dim, nclasses, wdecay)
optTheta=sm.train(data, labels, 100) # max iteration 100


#===============================================================================
## [Release Test -- testing]

## load optTheta
mat_file = loadmat('../softmax_release_test_TestingPhase.mat',squeeze_me=False)
# optTheta=mat_file['optTheta']
testingData=mat_file['inputData']
testingLabels=mat_file['labels']
testingLabels=testingLabels-1  # ! convert to zero-indexing

nclasses = 10
dim = testingData.shape[0]

print "=========================\n input Parms"
print "testingData"
print testingData.shape
print "testing labels: "
print testingLabels.shape
print "dim:"
print dim
print "nclasses:"
print nclasses
print "========================="

pred=softmax_predict(optTheta, nclasses, dim, testingData)






