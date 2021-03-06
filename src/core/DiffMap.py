from numpy import shape, mat, kron, exp, zeros, ones, divide, dot, diag, transpose, sqrt
from scipy.sparse.linalg import eigsh as eigsh
from kernel import kernel
# import scipy
class DiffMap:
    """
    Implements a class for diffusion maps. 
 
    Example usage::

        # load data
        mat_file = loadmat('example.mat',squeeze_me=False) # load data, and use matplotlib to plot it
        data = np.transpose(mat_file['data'])
 
        # run diffusion map 
        neigs = 3 # rank of embedding
        t_parm = 4 # 'time' for random walk
        dmap_obj = DiffMap(k,neigs)
        dmap_obj.process(data) # build map
        coeff = kpca_obj.reduce(data) # project data onto space using Nystrom extension

    :param k_params: Kernel parameters
    :type k_params: numpy mat 
    :return: `DiffMap object`
    :rtype:  DiffMap
    """
    def __init__(self, k_type, neigs):
        """
        This initialization function stores the kernel and number of 
        eigenvectors to retain. 
        """
        # need to check if parameters passed are legitimate
        k_name = k_type.name
        k_params = k_type.params  
        if k_name == "polynomial": 
            k_size = k_params.shape
            if k_size[0] != 2:
                print "Wrong number of parameters: polynomial\
                       kernel needs degree and bias\n"
        elif k_name == "gaussian":
            k_size = k_params.shape
            if k_size[0] != 1:
                print "Wrong number of parameters: gaussian\
                       kernel needs bandwidth\n"      
        elif k_name == "sigmoid": 
            k_size = k_params.shape
            if k_size[0] != 2:
                print "Wrong number of parameters: sigmoid\
                       kernel needs bandwidth\n"            
        else:
            print "Invalid kernel type: supported types are\
                    polynomial, gaussian, sigmoid\n"

        # now safe to initialize
        self.kernel = k_type
        self.neigs = neigs

    def process(self, data):
        """
        This function builds the diffusion matrix from the data, performs the random walk, and 
        returns the diffused data points. 

        :param data: data matrix :math:`D\in\mathbb{R}^{d \\times n}`, where 
                     :math:`d` is the dimensionality and 
                     :math:`n` is the number of training samples
        :type data: numpy array
        :return: diffused data
        :rtype:  numpy array
        """        
        # perform eigendecomposition
        k_mat = kernel(data, data, self.kernel)                
        k_evals, k_evecs = eigsh(k_mat, k=self.neigs, which='LM') 
        k_evals = sqrt(k_evals)
	k_scale = (ones((1,self.neigs),float)/k_evals)
	k_scale = diag(k_scale[0,:]) # create scaling matrix 
        alpha = dot(k_evecs, k_scale)

	self.alpha = alpha # store copies of coefficients and data
	self.data = data

    def project(self, tdata):
        """
        This function uses the Nystrom extension to projects the test data onto 
        the diffusion space. 

        :param tdata: data matrix :math:`D\in\mathbb{R}^{d \\times m}`, where 
                     :math:`d` is the dimensionality and 
                     :math:`m` is the number of testing samples
        :type tdata: numpy array
        :return: projected test data
        :rtype:  numpy array
        """        
	K_test = kernel(self.data, tdata, self.kernel)	
	coeff = dot(transpose(self.alpha), K_test)
	return coeff



