# this script tests KernelType and kernel 

# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# matplotlib stuff
import matplotlib as mp
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# add class directory to path
import sys
sys.path.insert(0, '../KMVis')

# our imports 
from kernel import *
from KernelType import *

k_name1   = "gaussian"
k_name2   = "sigmoid"
k_name3   = "polynomial"
k_name4   = "wrongname"
k_params1 = np.array( [1.2] )
k_params2 = np.array( [0.5, 1.2] )
k_params3 = np.array( [2, 0] )

k1 = KernelType(k_name1,k_params1)
k2 = KernelType(k_name2,k_params2)
k3 = KernelType(k_name3,k_params3)
k4 = KernelType(k_name4,k_params3)


# check names etc
print "Kernel 1:"
print "Name: "  + k1.name
print "Parms: " 
print  k1.params

print "\nKernel 2:"
print "Name: "  + k2.name
print "Parms: " 
print  k2.params

print "\nKernel 3:"
print "Name: "  + k3.name
print "Parms: " 
print  k3.params

# now, generate plots for kernels 
x = np.arange(-5,5,0.1)
y = np.array([2]) # y vals

k_gauss = kernel(x,y,k1)
k_sigm = kernel(x,y,k2)
k_poly = kernel(x,y,k3)

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

plt.draw()
plt.show()




