# test SMO

# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# matplotlib stuff
from matplotlib import rc
import matplotlib.pyplot as plt


# add class directory to path
from time import time
import sys
sys.path.insert(0, '../src')

# our imports 
from kernel import *
from KernelType import *
from KernelSVM import *

# load data, and use matplotlib to plot it
data_file = loadmat('../data/kernel_trick_data.mat',squeeze_me=False)
classes_file = loadmat('../data/kernel_trick_classes.mat',squeeze_me=False)

data1 = data_file['x1']
data2 = data_file['x2']
labels1 = classes_file['c1']
labels2 = classes_file['c2']


#data1 = np.array([[-2, -1],[-2,-1]])
#data2 = np.array([[2, 1],[2,1]])
#labels1 = -1*np.ones((1,2))
#labels2 = np.ones((1,2))

# initialize kernel
k_name   = "gaussian"
k_params = np.array( [2] )
#k_name   = "sigmoid"
#k_params = np.array( [1,0.1] )
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


plt.draw()
plt.show()










