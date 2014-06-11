# test SMO

# ensure reproducible behavior
import random

# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# matplotlib stuff
from matplotlib import rc
import matplotlib.pyplot as plt


# add class directory to path
from time import time
import sys, os
sys.path.insert(0, '../src/core')
sys.path.insert(0, '../data/examples')
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.KernelSVM import KernelSVM

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

# boolean which tells program to save output so it
# can be used by unit testing framework
save_data = False

# set random seed to ensure reproducible pseudorandom behavior
np.random.seed(3)

# load data, and use matplotlib to plot it
ksvm_data_filename = 'kernel_trick_data.mat'
ksvm_data_filepath = genloadstring(data_path,ksvm_data_filename)
data_file = loadmat(ksvm_data_filepath,squeeze_me=False)

ksvm_classes_filename = 'kernel_trick_classes.mat'
ksvm_classes_filepath = genloadstring(data_path,ksvm_classes_filename)
classes_file = loadmat(ksvm_classes_filepath,squeeze_me=False)

data1 = data_file['x1']
data2 = data_file['x2']
labels1 = classes_file['c1']
labels2 = classes_file['c2']

# initialize kernel
k_name   = "gaussian"
k_params = np.array( [2] )
k = KernelType(k_name,k_params)

# initialize parameters
C = 100
tol = 0.001
max_its = 100

# initialize SVM object
start = time()
ksvm = KernelSVM(k,C,tol,max_its)

# prepare, and pass in data
data = np.hstack([data1,data2])
labels = np.hstack([labels1,labels2])
ksvm.process(data,labels)
elapsed = (time() - start)
print "Gaussian SVM training time:", elapsed, "seconds"

# call SMO
#alpha, b = simplesmo(data,classes,k,C,tol,max_its)

# compute accuracy
ntest = data.shape[1]
predicted_labels = ksvm.predict(data)
diff = np.absolute(labels - predicted_labels)
err = np.sum(diff)/(2*ntest)
print "Gaussian SVM training error:", err

# compute boundary
start = time()
xv, yv, vals = ksvm.evaluateboundary(data,100,100)
elapsed = (time() - start)
print "Boundary evaluation time:", elapsed, "seconds"

# plot original data and boundary contour
fig = plt.figure()
ax = fig.gca()
ax.plot(data1[0,:],data1[1,:],'bo')
ax.plot(data2[0,:],data2[1,:],'ro')
ax.contour(xv,yv,vals, np.array([0, 0])) # plot countour at middle
ax.set_title(r"Nonlinearly separable data",fontsize=20)

if save_data:
    savemat(test_path + '/' + 'test_ksvm.mat',
            {'xv':xv,'yv':yv,'vals':vals,
             'predicted_labels':predicted_labels})

plt.draw()
plt.show()
