# this script tests the MeanMap class

# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# matplotlib stuff
import matplotlib as mp
from matplotlib import rc
import matplotlib.pyplot as plt

# add class directory to path
import sys, os

# do imports 
from ..src.core.KernelType import KernelType
from ..src.core.MeanMap import MeanMap

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))


# boolean which tells program to save output so it
# can be used by unit testing framework
save_data = False

# turns on Tex for plotting
rc('text', usetex=True)
rc('font', family='serif')

# load data
mmd_data_filename = 'mmd_data.mat'
mmd_data_filepath = genloadstring(data_path,mmd_data_filename)
mmd_mat_file = loadmat(mmd_data_filepath,squeeze_me=False)
data1 = mmd_mat_file['data1'] # data from distribution 1
data2 = mmd_mat_file['data2'] # data from distribution 1
data3 = mmd_mat_file['data3'] # data from distribution 2

# plot original data
fig = plt.figure()
ax = fig.gca()
p1, = ax.plot(data1[0,:],data1[1,:],'ro')
p2, = ax.plot(data2[0,:],data2[1,:],'bo')
p3, = ax.plot(data3[0,:],data3[1,:],'go')
plt.legend([p1, p2, p3], ["samples from $p_1(x)$", "samples from $p_2(x)$","samples from $p_3(x)$"])
ax.set_title(r"Samples from Two Probability Distributions",fontsize=20)

# set up kernel
k_name = "gaussian"
k_params = np.array( [3] ) # numpy array
k = KernelType(k_name, k_params)

# initialize MeanMap
mm_obj = MeanMap(k)
mm_obj.process(data1) # build map

# compute maximum mean discrepancy between data1 and data2
dist1 = mm_obj.mmd(data2)
dist2 = mm_obj.mmd(data3)
print "The MMD between data1 and data2 is " + str(dist1)
print "The MMD between data1 and data3 is " + str(dist2)

if save_data:
    savemat(test_path + '/' + 'test_mmd.mat',
            {'dist1':dist1,'dist2':dist2})

plt.draw()
plt.show()

