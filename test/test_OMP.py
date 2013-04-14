# this script tests the Orthogonal Matching Pursuit algorithm
# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat

# add class directory to path
import sys
sys.path.insert(0, '../KMVis')

# our imports 
from time import time
from OMP import *

# load data, and use matplotlib to plot it
mat_file = loadmat('../data/OMP_test.mat',squeeze_me=False)
data = mat_file['data']
obs = mat_file['obs']

start = time()
omp = OMP(-1,-1)
omp.process(data,obs)
elapsed = (time() - start)
print "Elapsed time:", elapsed, "seconds"

