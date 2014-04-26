# this script tests kernel ridge regression
# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# matplotlib stuff
import matplotlib as mp
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# add class directory to path
import sys
sys.path.insert(0, '../src/core')
sys.path.insert(0, '../data/examples')

# our imports
from time import time
from KernelRidgeRegression import *
from kernel import *
from KernelType import *

# boolean which tells program to save output so it
# can be used by unit testing framework
save_data = True;

# load data, and use matplotlib to plot it
mat_file = loadmat('KRR_test.mat',squeeze_me=False)
data = mat_file['x']
obs = mat_file['y_n']

k_name   = "gaussian"
k_params = np.array( [0.5] )
k = KernelType(k_name,k_params)

start = time()
krr1 = KernelRidgeRegression(k, -1)
krr2 = KernelRidgeRegression(k, 0.2)
krr1.process(data,obs)
krr2.process(data,obs)

est_obs1 = krr1.predict(data)
est_obs2 = krr2.predict(data)

# clean up
est_obs1 = np.squeeze(np.asarray(est_obs1))
est_obs2 = np.squeeze(np.asarray(est_obs2))

# plot fits
#data = np.mat(data)
#obs = np.mat(obs)

# turns on Tex
rc('text', usetex=True)
rc('font', family='serif')

fig = plt.figure()
ax = fig.gca()
p1, = ax.plot(data[0,:],obs[0,:],'ro')
p2, = ax.plot(data[0,:],est_obs1,'b', linewidth=2.5)
ax.set_title(r"Kernel Regression",fontsize=20)
plt.legend([p1, p2], ["data", "non-regularized"])

fig2 = plt.figure()
ax2 = fig2.gca()
p1, = ax2.plot(data[0,:],obs[0,:],'ro')
p3, = ax2.plot(data[0,:],est_obs2,'g', linewidth=2.5)
ax2.set_title(r"Kernel Ridge Regression",fontsize=20)
plt.legend([p1, p3], ["data", "regularized"])

if save_data:
    savemat('../data/unit_tests/test_krr.mat',
            {'est_obs1':est_obs1,'est_obs2':est_obs2})

plt.draw()
plt.show()