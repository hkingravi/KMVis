""" Import modules necessary for KPCA class. """
from numpy import ones, diag, transpose, sqrt, eye, tile, dot
from scipy.sparse.linalg import eigsh as eigsh
from kernel import kernel
# import scipy
class KPCA:
    """
    Implements a class performing kernel PCA.

    Example usage::

        # load data
        mat_file = loadmat('example.mat', squeeze_me=False) # load data, and use matplotlib to plot it
        data = np.transpose(mat_file['data'])

        # set up kernel
        k_name   = "gaussian"
        k_params = np.array( [1.2] )
        k = KernelType(k_name,k_params)

        # run kpca
        neigs = 3
        centered = 0
        kpca_obj = KPCA(k, neigs, centered)
        kpca_obj.process(data) # generate eigenspace
        coeff = kpca_obj.project(data) # project data onto eigenspace

    :param k_type: Kernel type
    :type k_type: KernelType
    :param neigs: Rank of eigendecomposition
    :type integer: integer
    :param centered: Parameter to define whether to center in feature space: 0 for no, 1 for yes.
    :type integer: integer
    :return: `KPCA object`
    :rtype:  KPCA
    """
    def __init__(self, k_type, neigs, centered):
        """
        This initialization function stores the kernel and number of
        eigenvectors to retain.
        """
        # safe to initialize, since KernelType does its own checking
        self.kernel = k_type
        self.neigs = neigs
        self.centered = centered

        # these member variables filled later
        self.k_cent = []
        self.alpha = []
        self.data = []
        self.h_mat = []

    def process(self, data):
        """
        This function builds the kernel matrix from the data, and generates the
        KPCA eigenspace.

        :param data: data matrix :math:`D\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of training samples
        :type data: numpy array
        """
        # perform eigendecomposition
        k_mat = kernel(data, data, self.kernel)

        # if centered, need to perform extra computations
        if self.centered > 0:
            nsamp = data.shape[1]
            unit = ones((nsamp, 1), float)
            self.h_mat = eye(nsamp) - (1./nsamp)*dot(unit, unit.T) # centering matrix
            self.k_cent = (1./nsamp)*dot(k_mat, unit) # centering vector, used in reduction

            k_mat = dot(k_mat, self.h_mat) # augmented kernel matrix
            k_mat = dot(self.h_mat, k_mat)

        k_evals, k_evecs = eigsh(k_mat, k=self.neigs, which='LM')
        k_evals = sqrt(k_evals)
        k_scale = (ones((1, self.neigs), float)/k_evals)
        k_scale = diag(k_scale[0, :]) # create scaling matrix
        alpha = dot(k_evecs, k_scale)

        self.alpha = alpha # store copies of coefficients and data
        self.data = data

    def project(self, tdata):
        """
        This function builds the kernel matrix from the data, and the test data,
        and projects the test data onto the KPCA eigenspace.

        :param tdata: data matrix :math:`D\in\mathbb{R}^{d \\times m}`, where
                     :math:`d` is the dimensionality and
                     :math:`m` is the number of testing samples
        :type tdata: numpy array
        :return: eigenfunction coefficients
        :rtype:  numpy array
        """
        k_test = kernel(self.data, tdata, self.kernel)

        if self.centered > 0:
            ntest = tdata.shape[1]
            k_test =  k_test - tile(self.k_cent, (1, ntest))
            k_test = dot(self.h_mat, k_test)

        coeff = dot(transpose(self.alpha), k_test)
        return coeff



