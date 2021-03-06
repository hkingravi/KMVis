# this script tests the Orthogonal Matching Pursuit algorithm
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
from time import time
import sys
sys.path.insert(0, '../src/core')
sys.path.insert(0, '../data/examples')

# our imports
from RidgeRegression import *

# load data, and use matplotlib to plot it
mat_file = loadmat('RR_test.mat',squeeze_me=False)
data = mat_file['x']
obs = mat_file['y']

start = time()
rr1 = RidgeRegression(-1)
rr2 = RidgeRegression(1)
rr1.process(data,obs)
rr2.process(data,obs)

est_obs1 = rr1.reduce(data)
est_obs2 = rr2.reduce(data)


# plot fits
rc('text', usetex=True)
rc('font', family='serif')


fig = plt.figure()
ax = fig.gca()
p1, = ax.plot(data[:,1],obs[:,0],'ro')
p2, = ax.plot(data[:,1],est_obs1[:,0],'b', linewidth=2.5)
ax.set_title(r"Linear Regression",fontsize=20)
plt.legend([p1, p2], ["data", "non-regularized"])

fig2 = plt.figure()
ax2 = fig2.gca()
p1, = ax2.plot(data[:,1],obs[:,0],'ro')
p3, = ax2.plot(data[:,1],est_obs2[:,0],'g', linewidth=2.5)
ax2.set_title(r"Ridge Regression",fontsize=20)
plt.legend([p1, p3], ["data", "non-regularized"])


plt.draw()
plt.show()