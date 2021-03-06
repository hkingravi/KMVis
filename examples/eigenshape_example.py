# this script tests the Eigenshape class

# numpy stuff
import numpy as np
from scipy.io import loadmat

# matplotlib stuff
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rc

# add class directory to path
from time import time
import sys
sys.path.insert(0, '../src/core')
sys.path.insert(0, '../data/examples')

# our imports
from EigenShape import *

# load data
mat_file = loadmat('yale_sub.mat',squeeze_me=False)
faces = np.transpose(mat_file['yale_sub_images'])
labels = np.asarray(mat_file['yale_sub_labels'])

start = time()
neigs = 20
es = Eigenshape(neigs)
es.process(faces)
eigenfaces = es.get("eigenshapes")

elapsed = (time() - start)
print "Elapsed time:", elapsed, "seconds"


# show some of the faces
width = 30
height = 40
indices = [0, 1, 2, 3, 4, 5, 6, 7]

# turns on Tex
rc('text', usetex=True)
rc('font', family='serif')

# example images
fig_orig_img = plt.figure()
for i in indices:
    img = faces[:,i]
    img = np.reshape(img, (width,height), order="F")
    plt.subplot(2, 4, i)
    imgplot = plt.imshow(np.float32(img),cmap = cm.Greys_r)

ax = fig_orig_img.gca()
plt.suptitle(r"Face Images",fontsize=20)

# see eigenfaces
fig_efaces = plt.figure()
for i in indices:
    eface = eigenfaces[:,i]
    eface = np.reshape(eface, (width,height), order="F")
    plt.subplot(2, 4, i)
    imgplot = plt.imshow(np.float32(eface),cmap = cm.Greys_r)

ax2 = fig_efaces.gca()
plt.suptitle(r"Eigenfaces",fontsize=20)

plt.show()