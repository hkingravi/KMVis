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
import sys, os

# our imports
from time import time
from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.KernelRidgeRegression import KernelRidgeRegression

from ..src.utils.genloadstring import genloadstring # for loading data 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

# boolean which tells program to save output so it
# can be used by unit testing framework
save_data = False

# load data, and use matplotlib to plot it
krr_data_filename = 'KRR_test.mat'
krr_data_filepath = genloadstring(data_path,krr_data_filename)
mat_file = loadmat(krr_data_filepath,squeeze_me=False)
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
    savemat(test_path + '/' + 'test_krr.mat',
            {'est_obs1':est_obs1,'est_obs2':est_obs2})

plt.draw()
plt.show()
