"""
Simple implementation of eigenshape concept. 
"""
from numpy import shape, dot, transpose, float64, argsort, squeeze, asarray
from scipy.sparse.linalg import eigsh
class Eigenshape:
    """
    Implements a class performing Eigenshapes. 
 
    Example usage: 
    >>> k_name = "sigmoid"
    >>> k_params = array( [0.5, 1.2] ) # numpy array
    >>> k = KernelType(k_name,k_params)

    :param k_name: Kernel name
    :type k_name: string
    :param k_params: Kernel parameters
    :type k_params: numpy mat 
    :return: `KernelType object`
    :rtype:  KernelType
    """
    def __init__(self, neigs):
        """
        This initialization function stores the kernel and number of 
        eigenvectors to retain. 
        """
        # need to check if parameters passed are legitimate
        if neigs < 1:             
            print "Incorrect number of retained eigenvectors:\
                   must be greater than 0."
        else:
	    # now safe to initialize
            self.neigs = neigs
            self.eigenshapes = []

    def process(self, data):
        """
        This function builds the covariance matrix from the data, and performs the 
        associated eigendecomposition.
        Data is a d x n numpy matrix, with d being the dimensionality, and n 
        being the number of samples. 
        """        
	# minus out average in data
        avg_data = data.mean(axis=1)
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] - avg_data
		
        # compute 'compact matrix' and do eigendecomposition
        data_dot = float64(dot(transpose(data), data))
        v_evals, v_evecs = eigsh(data_dot, k=self.neigs, which='LM')

	# need to sort eigenvectors by greatest eigenvalue
        indices = asarray(argsort(v_evals))
        indices = indices[::-1] # reverse order
        v_evals = squeeze(v_evals[indices])
        v_evecs = squeeze(v_evecs[:, indices])

	# finally, generate eigenshapes
        self.eigenshapes = dot(data, v_evecs)

    def get(self,field):
        """
        This function returns the two field instances of the object. 
        """  
        if field == "neigs":
            return self.neigs
        elif field == "eigenshapes":
            return self.eigenshapes
        else:
            print "Incorrect field."

