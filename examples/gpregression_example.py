# this script tests Bayesian RBF networks

# numpy stuff
import numpy as np
from scipy.io import loadmat, savemat
from numpy.linalg import svd
from numpy.linalg import norm

# matplotlib stuff
from matplotlib import rc
import matplotlib.pyplot as plt

# add class and data directories to path
import sys, os 

from ..src.core.KernelType import KernelType
from ..src.core.kernel import kernel
from ..src.core.BayesianRBF import BayesianRBF

from ..src.utils.genloadstring import genloadstring # for loading data 
from ..src.utils.gencolorarray import gencolorarray # for coloring plot 
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/unit_tests'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/examples'))

# boolean which tells program to save output so it
# can be used by unit testing framework
save_data = True

# turns on Tex for plotting
rc('text', usetex=True)
rc('font', family='serif')

# set random seed to ensure reproducible pseudorandom behavior
np.random.seed(3)

# plotting parameters
func_lwidth = 1.5
basis_lwidth = 2.5

# load data, and use matplotlib to plot it
brbf_data_filename = 'BRBF_test.mat'
brbf_data_filepath = genloadstring(data_path,brbf_data_filename)
brbf_mat_file = loadmat(brbf_data_filepath,squeeze_me=False)
data = brbf_mat_file['data']
obs = brbf_mat_file['cvals']
centers = brbf_mat_file['centers']
eval_data = brbf_mat_file['eval_data']
evals = brbf_mat_file['evals']

# first, plot the data
fig = plt.figure()
ax = fig.gca()
p1, = ax.plot(eval_data[0,:],evals[0,:],'g')
p2, = ax.plot(data[0,:],obs[0,:],'bo', linewidth=2.5)
ax.set_title(r"Model Data",fontsize=20)
plt.legend([p1, p2], ["function", "samples"])

k_name   = "gaussian"
k_params = np.array( [0.5] )
k = KernelType(k_name,k_params)

brbf1 = BayesianRBF(k, 0.2, centers)
brbf1.process(data,obs)

f, var_f = brbf1.predict(data)

# plot random functions
num_rfunc = 40
ntest = eval_data.shape[1]

fig2 = plt.figure()
ax2 = fig2.gca()
A = np.zeros((num_rfunc,ntest))

for i in xrange(0, num_rfunc):
    # draw random function
    f_rand = brbf1.draw_rfunc(eval_data)
    ax2.plot(eval_data[0,:],f_rand,'r',linewidth=func_lwidth)

    # to save f_rand, you need to reshape appropriately
    f_rand = np.asarray(f_rand)
    f_rand = f_rand.reshape(1,ntest)
    A[i,:] = f_rand

plt.title(r"Random functions",fontsize=20)

# now, generate basis for random functions using SVD
rank1 = 1
rank2 = 3
rank3 = 8
U, S, V = svd(A, full_matrices=True)

# create low-rank matrices
S = np.diag(S)
U_ret1 = U[:,0:rank1]
V_ret1 = V[0:rank1,:]
S_ret1 = S[0:rank1,0:rank1]

U_ret2 = U[:,0:rank2]
V_ret2 = V[0:rank2,:]
S_ret2 = S[0:rank2,0:rank2]

U_ret3 = U[:,0:rank3]
V_ret3 = V[0:rank3,:]
S_ret3 = S[0:rank3,0:rank3]

# reconstruct
A_r1 = np.dot(U_ret1,np.dot(S_ret1,V_ret1))
A_r2 = np.dot(U_ret2,np.dot(S_ret2,V_ret2))
A_r3 = np.dot(U_ret3,np.dot(S_ret3,V_ret3))

if save_data:
    savemat(test_path + '/' + 'test_brbf.mat',
            {'A_r1':A_r1,'A_r2':A_r2,'A_r3':A_r3})

# plot reconstructed signals for rank1
fig3a = plt.figure()
ax3a = fig3a.gca()

for i in xrange(0, num_rfunc):
    f_recon = A_r1[i,:]
    ax3a.plot(eval_data[0,:],f_recon,'r',linewidth=func_lwidth)

plt.title(r"Reconstructed functions: Rank 1",fontsize=20)

fig3b = plt.figure()
ax3b = fig3b.gca()

color_list = gencolorarray(rank1) # get a list of colors for plotting
for i in xrange(0, rank1):
    fe = V_ret1[i,:]/norm(V_ret1[i,:])
    ax3b.plot(eval_data[0,:],fe, color= color_list[i],linewidth=basis_lwidth,linestyle='--')
plt.title(r"Eigenbasis: Rank 1",fontsize=20)

# plot reconstructed signals for rank2
fig4a = plt.figure()
ax4a = fig4a.gca()

for i in xrange(0, num_rfunc):
    f_recon = A_r2[i,:]
    ax4a.plot(eval_data[0,:],f_recon,'r',linewidth=func_lwidth)

plt.title(r"Reconstructed functions: Rank 3",fontsize=20)

fig4b = plt.figure()
ax4b = fig4b.gca()

color_list = gencolorarray(rank2) # get a list of colors for plotting
for i in xrange(0, rank2):
    fe = V_ret2[i,:]/norm(V_ret2[i,:])
    ax4b.plot(eval_data[0,:],fe, color= color_list[i],linewidth=basis_lwidth,linestyle='--')
plt.title(r"Eigenbasis: Rank 3",fontsize=20)


# plot reconstructed signals for rank3
fig5a = plt.figure()
ax5a = fig5a.gca()

for i in xrange(0, num_rfunc):
    f_recon = A_r3[i,:]
    ax5a.plot(eval_data[0,:],f_recon,'r',linewidth=func_lwidth)

plt.title(r"Reconstructed functions: Rank 8",fontsize=20)

fig5b = plt.figure()
ax5b = fig5b.gca()

color_list = gencolorarray(rank3) # get a list of colors for plotting
for i in xrange(0, rank3):
    fe = V_ret3[i,:]/norm(V_ret3[i,:])
    ax5b.plot(eval_data[0,:],fe, color= color_list[i],linewidth=basis_lwidth,linestyle='--')
plt.title(r"Eigenbasis: Rank 8",fontsize=20)

plt.draw()
plt.show()
