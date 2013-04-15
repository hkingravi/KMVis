# this script just tests the KernelType function to see 
# whether the initializations are working

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
sys.path.insert(0, '../KMVis')

# our imports 
from kernel import *
from KernelType import *
from KPCA import *

np.set_printoptions(precision=4)

k_name1   = "gaussian"
k_name2   = "sigmoid"
k_name3   = "polynomial"

k_params1 = np.array( [10] )
k_params2 = np.array( [0.5, 1.2] )
k_params3 = np.array( [2, 0] )

k1 = KernelType(k_name1,k_params1)
k2 = KernelType(k_name2,k_params2)
k3 = KernelType(k_name3,k_params3)

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

# load data, and use matplotlib to plot it
mat_file = loadmat('../data/separable_gmm.mat',squeeze_me=False)

data = np.transpose(mat_file['data'])
labels = mat_file['labels']

# get and clean labels of data
l1 = np.asarray(np.nonzero(labels == 1))
l2 = np.asarray(np.nonzero(labels == 2))
l3 = np.asarray(np.nonzero(labels == 3))
l4 = np.asarray(np.nonzero(labels == 4))

l1 = l1[0,:]
l2 = l2[0,:]
l3 = l3[0,:]
l4 = l4[0,:]

print "\nData dimensionality: ", data.shape

# run KPCA, and compute eigenmbedding of data
start = time()
neigs = 3 
centered = 1
kpca_obj = KPCA(k1, neigs, centered)
kpca_obj.process(data)
coeff = kpca_obj.reduce(data)
elapsed = (time() - start)
print "Gaussian KPCA Elapsed time:", elapsed, "seconds"


# turns on Tex
rc('text', usetex=True)
rc('font', family='serif')

# plot original data 
fig = plt.figure()
ax = fig.gca()
ax.plot(data[0,l1],data[1,l1],'ro')
ax.plot(data[0,l2],data[1,l2],'go')
ax.plot(data[0,l3],data[1,l3],'bo')
ax.plot(data[0,l4],data[1,l4],'ko')
ax.set_title(r"Data generated from Gaussian Mixture Models",fontsize=20)

# plot embedded data 
coeff1 = np.squeeze(np.asarray(coeff[:,l1]))
coeff2 = np.squeeze(np.asarray(coeff[:,l2]))
coeff3 = np.squeeze(np.asarray(coeff[:,l3]))
coeff4 = np.squeeze(np.asarray(coeff[:,l4]))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(coeff1[0,:], coeff1[1,:], coeff1[2,:], 'ro')
ax2.plot(coeff2[0,:], coeff2[1,:], coeff2[2,:], 'go')
ax2.plot(coeff3[0,:], coeff3[1,:], coeff3[2,:], 'bo')
ax2.plot(coeff4[0,:], coeff4[1,:], coeff4[2,:], 'ko')
ax2.set_title(r"Gaussian KPCA Embedding for GMM Data",fontsize=20)

# now try polynomial kernel data
mat_file = loadmat('../data/kernel_trick_data.mat',squeeze_me=False)

polydata1 = mat_file['x1']
polydata2 = mat_file['x2']
polydata = hstack([polydata1,polydata2])

print "\nData dimensionality: ", polydata.shape
start = time()
neigs = 3 
kpca_obj2 = KPCA(k3, neigs, centered)
kpca_obj2.process(polydata)
polycoeff1 = kpca_obj2.reduce(polydata1)
polycoeff2 = kpca_obj2.reduce(polydata2)
elapsed = (time() - start)
print "Poly KPCA Elapsed time:", elapsed, "seconds"

polycoeff1 = np.squeeze(np.asarray(polycoeff1))
polycoeff2 = np.squeeze(np.asarray(polycoeff2))

# plot polynomial kernel data
fig3 = plt.figure()
ax3 = fig3.gca()
ax3.plot(polydata1[0,:],polydata1[1,:],'ro')
ax3.plot(polydata2[0,:],polydata2[1,:],'bo')
ax3.set_title(r"Non-linearly separable data",fontsize=20)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot(polycoeff1[0,:], polycoeff1[1,:], polycoeff1[2,:], 'ro')
ax4.plot(polycoeff2[0,:], polycoeff2[1,:], polycoeff2[2,:], 'bo')
ax4.set_title(r"Polynomial KPCA Embedding for Nonseperable Data",fontsize=20)


# now plot actual eigenfunctions of both Gaussian and polynomial KPCA
# first, need to get minimum and maximum from the data along both axes
GMM_x_vals = data[0,:]
GMM_y_vals = data[1,:]
x_min = np.amin(GMM_x_vals)
y_min = np.amin(GMM_y_vals)
x_max = np.amax(GMM_x_vals)
y_max = np.amax(GMM_y_vals)

# generate meshgrid
X = np.arange(x_min, x_max, 0.1).tolist()
Y = np.arange(y_min, y_max, 0.1).tolist()
X, Y = np.meshgrid(X, Y)
Z0 = np.zeros((X.shape[0],X.shape[1]),float) # temp
Z1 = np.zeros((X.shape[0],X.shape[1]),float) # temp
Z2 = np.zeros((X.shape[0],X.shape[1]),float) # temp

start = time()
for i in range(len(X[0])):
   for j in range(len(X)):
        coeff = kpca_obj.reduce(np.vstack((X[j][i],Y[j][i])))
        Z0[j][i] = np.squeeze(np.asarray(coeff[0]))
        Z1[j][i] = np.squeeze(np.asarray(coeff[1]))
        Z2[j][i] = np.squeeze(np.asarray(coeff[2]))

elapsed = (time() - start)
print "Gaussian surface plot generation time:", elapsed, "seconds"

# polynomial
poly_x_vals = polydata[0,:]
poly_y_vals = polydata[1,:]
poly_x_min = np.amin(poly_x_vals)
poly_y_min = np.amin(poly_y_vals)
poly_x_max = np.amax(poly_x_vals)
poly_y_max = np.amax(poly_y_vals)

# generate meshgrid
poly_X = np.arange(poly_x_min, poly_x_max, 0.02).tolist()
poly_Y = np.arange(poly_y_min, poly_y_max, 0.02).tolist()
poly_X, poly_Y = np.meshgrid(poly_X, poly_Y)
poly_Z0 = np.zeros((poly_X.shape[0],poly_X.shape[1]),float) # temp
poly_Z1 = np.zeros((poly_X.shape[0],poly_X.shape[1]),float) # temp
poly_Z2 = np.zeros((poly_X.shape[0],poly_X.shape[1]),float) # temp

start = time()
for i in range(len(poly_X[0])):
   for j in range(len(poly_X)):
        poly_coeff = kpca_obj2.reduce(np.vstack((poly_X[j][i],poly_Y[j][i])))
        poly_Z0[j][i] = np.squeeze(np.asarray(poly_coeff[0]))
        poly_Z1[j][i] = np.squeeze(np.asarray(poly_coeff[1]))
        poly_Z2[j][i] = np.squeeze(np.asarray(poly_coeff[2]))

elapsed = (time() - start)
print "Polynomial surface plot generation time:", elapsed, "seconds"

# GAUSSIAN
fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
#CS = plt.contour(X, Y, Z0, linewidths=2)
CS = ax5.plot_surface(X,Y,Z0,cmap=cm.jet)
plt.clabel(CS, inline=1, fontsize=10)
plt.title(r"Gaussian KPCA Eigenfunction 1 (Centered)",fontsize=20)

fig6 = plt.figure()
ax6 = fig6.gca(projection='3d')
CS1 = ax6.plot_surface(X,Y,Z1,cmap=cm.jet)
plt.clabel(CS1, inline=1, fontsize=10)
plt.title(r"Gaussian KPCA Eigenfunction 2 (Centered)",fontsize=20)

fig7 = plt.figure()
ax7 = fig7.gca(projection='3d')
CS2 = ax7.plot_surface(X,Y,Z2,cmap=cm.jet)
plt.clabel(CS2, inline=1, fontsize=10)
plt.title(r"Gaussian KPCA Eigenfunction 3 (Centered)",fontsize=20)


# POLYNOMIAL
fig8 = plt.figure()
ax8 = fig8.gca(projection='3d')
#CS = plt.contour(X, Y, Z0, linewidths=2)
poly_CS = ax8.plot_surface(poly_X,poly_Y,poly_Z0,cmap=cm.jet)
plt.clabel(poly_CS, inline=1, fontsize=10)
plt.title(r"Polynomial KPCA Eigenfunction 1 (Centered)",fontsize=20)

fig9 = plt.figure()
ax9 = fig9.gca(projection='3d')
poly_CS1 = ax9.plot_surface(poly_X,poly_Y,poly_Z1,cmap=cm.jet)
plt.clabel(poly_CS1, inline=1, fontsize=10)
plt.title(r"Polynomial KPCA Eigenfunction 2 (Centered)",fontsize=20)

fig10 = plt.figure()
ax10 = fig10.gca(projection='3d')
poly_CS2 = ax10.plot_surface(poly_X,poly_Y,poly_Z2,cmap=cm.jet)
plt.clabel(poly_CS2, inline=1, fontsize=10)
plt.title(r"Polynomial KPCA Eigenfunction 3 (Centered)",fontsize=20)


plt.draw()
plt.show()
















