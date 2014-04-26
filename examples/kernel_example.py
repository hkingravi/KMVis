# this script tests KernelType and kernel

# numpy stuff
import numpy as np
from scipy.io import savemat

# matplotlib stuff
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# add class directory to path
import sys
sys.path.insert(0, '../src/core')
sys.path.insert(0, '../src/exceptions')
sys.path.insert(0, '../data/examples')

# our imports
from kernel import kernel
from KernelType import KernelType
from Error import *

# boolean which tells program to save output so it
# can be used by unit testing framework
save_data = False;

# initialize kernels
k_name1   = "gaussian"
k_name2   = "sigmoid"
k_name3   = "polynomial"
k_name4   = "laplacian"
k_name5   = "cauchy"
k_name6   = "wrongname"
k_name7   = "periodic"
k_name8   = "locally_periodic"
k_params1 = np.array( [1.2] )
k_params2 = np.array( [0.5, 1.2] )
k_params3 = np.array( [2, 0] )
k_params4 = np.array( [1.2] )
k_params5 = np.array( [1.2] )
k_params7 = np.array( [1.2, 0.5] )
k_params8 = np.array( [0.5, 1] )

try:
    k1 = KernelType(k_name1,k_params1)
except Exception, e:
    print e.args
try:
    k2 = KernelType(k_name2,k_params2)
except Exception, e:
    print e.args
try:
    k3 = KernelType(k_name3,k_params3)
except Exception, e:
    print e.args
try:
    k4 = KernelType(k_name4,k_params4)
except Exception, e:
    print e.args
try:
    k5 = KernelType(k_name5,k_params5)
except Exception, e:
    print e.args
try:
    k6 = KernelType(k_name6,k_params5)
except KernelTypeError, e:
    s = str(e)
    print s
try:
    k7 = KernelType(k_name7,k_params7)
except Exception, e:
    print e.args
try:
    k8 = KernelType(k_name8,k_params8)
except Exception, e:
    print e.args


# print the names of the kernels
print k1
print k2
print k3
print k4
print k5
print k7

# now, generate plots for kernels
x = np.arange(-5,5,0.1)
x_rad = np.arange(-3,7,0.1)
y = np.array([2]) # y vals

k_gauss = kernel(x_rad,y,k1)
k_sigm = kernel(x,y,k2)
k_poly = kernel(x,y,k3)
k_lap = kernel(x_rad,y,k4)
k_cauchy = kernel(x_rad,y,k5)
k_periodic = kernel(x_rad,y,k7)
k_locally_periodic = kernel(x_rad,y,k8)

# save files in test data directory

if save_data:
    savemat('../data/unit_tests/test_kernel.mat',
            {'k_gauss':k_gauss,'k_sigm':k_sigm,'k_poly':k_poly,'k_lap':k_lap,
             'k_cauchy':k_cauchy,'k_periodic':k_periodic,'k_locally_periodic':k_locally_periodic})

# plot Gaussian kernel values

# turns on Tex
rc('text', usetex=True)
rc('font', family='serif')

fig = plt.figure()
ax = fig.gca()
ax.plot(x,k_gauss,'b', linewidth=2.5)
ax.set_title(r"Gaussian kernel with $\sigma = 1.2$",fontsize=20)

fig2 = plt.figure()
ax2 = fig2.gca()
ax2.plot(x,k_sigm,'b', linewidth=2.5)
ax2.set_title(r"Sigmoid kernel with $\alpha = 0.5, c = 1.2$",fontsize=20)

fig3 = plt.figure()
ax3 = fig3.gca()
ax3.plot(x,k_poly,'b', linewidth=2.5)
ax3.set_title(r"Polynomial kernel with $\gamma = 2, c = 0$",fontsize=20)

fig = plt.figure()
ax = fig.gca()
ax.plot(x,k_lap,'b', linewidth=2.5)
ax.set_title(r"Laplacian kernel with $\sigma = 1.2$",fontsize=20)

fig = plt.figure()
ax = fig.gca()
ax.plot(x,k_cauchy,'b', linewidth=2.5)
ax.set_title(r"Cauchy kernel with $\sigma = 1.2$",fontsize=20)

fig = plt.figure()
ax = fig.gca()
ax.plot(x,k_periodic,'b', linewidth=2.5)
ax.set_title(r"Periodic kernel with $\sigma = 1.2$, $p = 0.5$",fontsize=20)

fig = plt.figure()
ax = fig.gca()
ax.plot(x,k_locally_periodic,'b', linewidth=2.5)
ax.set_title(r"Locally Periodic kernel with $\sigma = 1.2$, $p = 0.5$",fontsize=20)

plt.draw()
plt.show()