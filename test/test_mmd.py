# this script tests the MeanMap class  

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
from MeanMap import *

# turns on Tex for plotting
rc('text', usetex=True)
rc('font', family='serif')

# load data
mat_file = loadmat('mmd_data.mat',squeeze_me=False) # load data, and use matplotlib to plot it
data1 = mat_file['data1'] # data from distribution 1
data2 = mat_file['data2'] # data from distribution 1
data3 = mat_file['data3'] # data from distribution 2

# plot original data 
fig = plt.figure()
ax = fig.gca()
p1, = ax.plot(data1[0,:],data1[1,:],'ro')
p2, = ax.plot(data2[0,:],data2[1,:],'bo')
p3, = ax.plot(data3[0,:],data3[1,:],'go')
plt.legend([p1, p2, p3], ["samples from $p_1(x)$", "samples from $p_2(x)$","samples from $p_1(x)$"])
ax.set_title(r"Samples from Two Probability Distributions",fontsize=20)

# set up kernel  
k_name = "gaussian"
k_params = np.array( [3] ) # numpy array
k = KernelType(k_name, k_params)        

# initialize MeanMap
mm_obj = MeanMap(k)
mm_obj.process(data1) # build map

dist1 = mm_obj.mmd(data2) 
dist2 = mm_obj.mmd(data3)

# compute maximum mean discrepancy between data1 and data2
print "The MMD between data1 and data2 is " + str(dist1)
print "The MMD between data1 and data3 is " + str(dist2)

plt.draw()
plt.show()




