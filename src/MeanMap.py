from numpy import sum as npsum
from kernel import kernel
class MeanMap:
    """
    Implements a class for computing the mean map :math:`\mu_{{D_1}}\in\mathcal{H}` 
    associated to an empirical sample :math:`D_1\in\mathbb{R}^{d \\times n}`, and
    a choice of kernel :math:`k(x,y)`. Recall that the mean map induces a metric in the space of 
    probability distributions convolved under the action of the kernel.The class also contains a 
    method to compute the maximum mean discrepancy between the mean map of :math:`D_1\in\mathbb{R}^{d \\times n}`
    and the mean map of :math:`D_2\in\mathbb{R}^{d \\times n}`.
 
    Example usage::

        # load data
        mat_file = loadmat('mmd_data.mat',squeeze_me=False) # load data, and use matplotlib to plot it
        data1 = np.transpose(mat_file['data1']) # data from distribution 1
        data2 = np.transpose(mat_file['data2']) # data from distribution 1
        data3 = np.transpose(mat_file['data3']) # data from distribution 2

        # set up kernel  
        k_name = "gaussian"
        k_params = array( [3] ) # numpy array
        k = KernelType(k_name, k_params)        
 
        # initialize MeanMap
        mm_obj = MeanMap(k)
        mm_obj.process(data) # build map

        # compute maximum mean discrepancy between data1 and data2
        dist1 = mm_obj.mmd(data2) 
        dist2 = mm_obj.mmd(data3) 
        print "The MMD between data1 and data2 is " + str(dist1)
        print "The MMD between data1 and data3 is " + str(dist2)

    :param k_type: Kernel 
    :type k_type: Kernel object
    :return: `MeanMap object`
    :rtype:  MeanMap
    """
    def __init__(self, k_type, *args):
        """
        This initialization function stores the kernel. 
        """
        self.kernel = k_type

    def process(self, data):
        """
        This function stores the data necessary for constructing 
        the mean map :math:`\mu_{{D_1}}\in\mathcal{H}`.

        :param data: data matrix :math:`D_1\in\mathbb{R}^{d \\times n}`, where 
                     :math:`d` is the dimensionality and 
                     :math:`n` is the number of training samples
        :type data: numpy array
        """        
        # parse data and ensure it meets requirements
        self.dim = data.shape[0]
        self.nsamp = data.shape[1]
        self.data = data

    def project(self, tdata):
        """
        This function projects test data :math:`D'` onto the mean map, i.e.
        it computes and returns :math:`\\langle\mu_{{D_1}},\\psi(x)\\rangle_\mathcal{H}`
        for each :math:`x\in D'`, where :math:`\\psi:\mathbb{R}^{d}\\to\mathcal{H}` is the feature
        map associated to the RKHS :math:`\mathcal{H}`. 

        :param data: testdata matrix :math:`D'\in\mathbb{R}^{d \\times n}`, where 
                     :math:`d` is the dimensionality and 
                     :math:`n` is the number of training samples
        :type data: numpy array
        :return: projected data
        :rtype:  numpy array
        """        
        # parse data and ensure it meets requirements
        test_dim = tdata.shape[0]
        ntest = tdata.shape[1]

        if test_dim != self.dim:
          raise Exception("ERROR: dimensionality of data must be consistent with training set.")

        # compute kernel matrix between data and tdata 
        kmat = kernel(self.data,tdata,self.kernel)
        kmat = npsum(kmat,axis=0)
        kmat = (1/self.nsamp)*kmat
        return kmat 
        
    def mmd(self, data2):
        """
        Compute the maximum mean discrepancy between 
        :math:`D_1` and :math:`D_2`, i.e. :math:`\\|\\mu_{{D_1}} - \\mu_{{D_2}}\\|_{\mathcal{H}}`

        :param data2: data matrix :math:`D_2\in\mathbb{R}^{d \\times n}`, where 
                     :math:`d` is the dimensionality and 
                     :math:`n` is the number of training samples
        :type data: numpy array
        :return: diffused data
        :rtype:  numpy array
        """        
        # parse data and ensure it meets requirements
        dim_data2 = data2.shape[0]
        ntest = data2.shape[1]                

        if dim_data2 != self.dim:
          raise Exception("ERROR: dimensionality of data must be consistent with training set.")

        # now, compute MMD equation, using in-place computations 
        kmat = kernel(data2,data2,self.kernel);
        ktestsum = npsum(npsum(kmat,axis=0),axis=1)/(pow(ntest,2));    

        kmat = kernel(self.data,self.data,self.kernel);
        ktrainsum = npsum(npsum(kmat,axis=0),axis=1)/(pow(self.nsamp,2));    

        kmat = kernel(self.data,data2,self.kernel);
        kcrossum = npsum(npsum(kmat,axis=0),axis=1)/(self.nsamp*ntest);                  

        mmdcomputed = ktestsum + ktrainsum - 2*kcrossum
 
        return mmdcomputed

