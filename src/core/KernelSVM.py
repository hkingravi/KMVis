from numpy import zeros, ones, dot, unique, sort, transpose, absolute, amax, amin, where
from numpy import linspace, meshgrid, hstack, concatenate
import numpy.random as rnd
from kernel import kernel
from math import fabs
class KernelSVM:
    """
    Implements a class for Kernel SVM. The SVM can utilize any of the kernels currently implemented
    in KernelType; however, it is the user's responsibility for determining whether a given kernel is
    positive-semidefinite. For example, the sigmoid kernel is only positive-semidefinite for certain
    parameters. If the kernel is not positive-semidefinite, the SMO algorithm may not converge.

    Example usage::

        # load data
        mat_file = loadmat('example.mat',squeeze_me=False) # load data, and use matplotlib to plot it
        data = np.transpose(mat_file['data'])

        # set up kernel
        k_name = "sigmoid"
        k_params = array( [0.5, 1.2] ) # numpy array
        k = KernelType(k_name, k_params)

        # initialize parameters for function optimization
        C = 10
        tol = 0.00001
        max_its = 100

        # run KernelSVM
        svm_obj = KernelSVM(k,neigs, C, tol, max_its)
        dmap_obj.process(data) # build map
        coeff = kpca_obj.reduce(data) # project data onto space using Nystrom extension

    :param k_params: Kernel parameters
    :type k_params: numpy mat
    :return: `KernelSVM object`
    :rtype:  KernelSVM
    """
    def __init__(self, k_type, *args):
        """
        This initialization function stores the kernel and initializes the
        parameters.
        """
        # need to check if parameters passed are legitimate
        self.kernel = k_type

        # create and set default parameters
        DEFAULT_C = 10
        DEFAULT_TOL = 0.0001
        DEFAULT_MAX_ITS = 1000
        self.C = DEFAULT_C
        self.tol = DEFAULT_TOL
        self.max_its = DEFAULT_MAX_ITS
        self.optimizer = 'simple'

        # overwrite parameters if not legitimate
        if (len(args) == 0):
            output_s = "Using default values for C, tolerance and max_iterations: " + str(self.C)
            output_s = output_s + ', ' + str(self.tol) + ', ' + str(self.max_its)
            print output_s

        if (len(args) >= 1):
            # make sure C is positive
            C = args[0]
            if C < 0:
                self.C = DEFAULT_C
                print "C must be positive: using default value of " + str(self.C)
            else:
                self.C = C

        if (len(args) >= 2):
            # make sure tolerance is positive
            tol = args[1]
            if tol < 0:
                self.tol = DEFAULT_TOL
                print "tolerance must be positive: using default value of " + str(self.tol)
            else:
                self.tol = tol

        if (len(args) >= 3):
            # make sure max_its is a positive integer
            max_its = args[2]
            if max_its <= 0:
                self.max_its = DEFAULT_MAX_ITS
                print "max_its must be positive: using default value of " + str(self.max_its)
            else:
                self.max_its = max_its

        # now you need to write out an appropriate message about the parameters used
        if (len(args) == 0):
            output_s = "Using default values for C, tolerance and max_iterations: " + str(self.C)
            output_s = output_s + ', ' + str(self.tol) + ', ' + str(self.max_its)
            print output_s
        elif (len(args) == 1):
            output_s = "Using default values for tolerance and max_iterations: "
            output_s = output_s + str(self.tol) + ', ' + str(self.max_its)
            print output_s
        elif (len(args) == 2):
            output_s = "Using default values for max_iterations: "
            output_s = output_s + str(self.max_its)
            print output_s

    def process(self, data, labels):
        """
        This function takes as input the data and labels from the user, and
        trains a kernel SVM.

        :param data: data matrix :math:`D\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of training samples
        :param labels: labels matrix :math:`Y\in\{-1,+1\}^{1 \\times n}`, where
                     :math:`n` is the number of training samples
        :type data: numpy array
        :return: diffused data
        :rtype:  numpy array
        """
        # parse data and ensure it meets requirements
        dim_data = data.shape[0]
        nsamp_data = data.shape[1]
        nsamp_labels = labels.shape[1]

        if nsamp_data != nsamp_labels:
            raise Exception("ERROR: number of samples for data and labels must be the same.")
        else:
            nsamp = nsamp_data

        label_vals = unique(labels)

        if label_vals.shape[0] != 2:
            raise Exception("ERROR: labels can only contain values -1 and +1.")
        else:
            # sort label_vals
            if label_vals[0] != -1 or label_vals[1] != 1:
                raise Exception("ERROR: labels can only contain values -1 and +1.")

        # create default weight vector, bias, support vectors and number of samples
        self.alpha = zeros((1,nsamp))
        self.bias = 0
        self.svecs = data
        self.slabels = labels
        self.nsamp = nsamp
        self.dim = dim_data

        # invoke optimization procedure specified
        if self.optimizer == 'simple':
            self.simplesmo(data,labels)

    def simplesmo(self, data, labels):
        """
        This function uses a simplified version of the SMO algorithm to train
        the weight and bias vectors associated to solution to the maximum margin
        problem. If the data is nonlinearly separable, this algorithm will not converge.
        See Andrew Ng's notes for more information.

        A technical point: in the SMO algorithm, we need to refer to elements in the
        numpy arrays. Therefore we have to use the item function.

        :param data: data matrix :math:`D\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of training samples
        :type data: numpy array
        """
        # initialize
        passes = 0
        alpha_old = zeros((1,self.nsamp))

        while (passes < self.max_its):
            num_changed_alphas = 0

            # process data
            for i in xrange(1,self.nsamp):
                # compute the Ei variable
                datai = self.svecs[:,i-1:i]
                labeli = (self.slabels).item(i-1)
                f_evali = self.evaluate(datai)
                f_evalj = f_evali.item(0)
                Ei = f_evali - labeli

                # compute conditions required
                alphai = (self.alpha).item(i-1)
                cond1 = (labeli*Ei < -self.tol and alphai < self.C)
                cond2 = (labeli*Ei > self.tol and alphai > 0)

                # check to see if you have to keep going; if you have to, you must optimize a lagrange multiplier pair
                if cond1 or cond2:
                    j = rnd.randint(1,self.nsamp,size=1)
                    j = j.item(0)

                    # compute eval_array[j]
                    dataj = self.svecs[:,j-1:j]
                    labelj = (self.slabels).item(j-1)
                    f_evalj = self.evaluate(dataj)
                    f_evalj = f_evali.item(0) # convert f_evali to scalar

                    Ej = f_evalj - labelj
                    alphaj = (self.alpha).item(j-1)

                    # save old Lagrange multipliers
                    alpha_old[:,i-1] = alphai
                    alpha_old[:,j-1] = alphaj

                    # compute L and H
                    if labeli != labelj:
                        L = amax([0,alphaj-alphai])
                        H = amin([self.C,self.C+alphaj-alphai])
                    else:
                        L = amax([0,alphai+alphaj-self.C])
                        H = amin([self.C,alphai+alphaj])

                    if L == H:
                        continue

                    # compute eta
                    eta = 2*kernel(datai,dataj,self.kernel) - kernel(datai,datai,self.kernel) - kernel(dataj,dataj,self.kernel)
                    eta = eta.item(0)

                    if eta >= 0:
                        continue

                    # compute new value for alphaj
                    alphaj = alphaj - (labelj*(Ei-Ej))/eta

                    # clip if necessary
                    if alphaj > H:
                        alphaj = H
                    elif alphaj < L:
                        alphaj = L

                    # if no change, continue
                    if (fabs(alphaj-alpha_old.item(j-1)) < self.tol):
                        continue

                    # otherwise, update alphai
                    self.alpha[:,i-1] = alphai + labeli*labelj*(alpha_old.item(j-1) - alphaj)

                    # compute threshold
                    b1 = self.bias - Ei - labeli*(alphai-alpha_old.item(i-1))*kernel(datai,datai,self.kernel)
                    b1 = b1 - labelj*(alphaj-alpha_old.item(j-1))*kernel(datai,dataj,self.kernel)

                    b2 = self.bias - Ej - labeli*(alphai-alpha_old.item(i-1))*kernel(datai,dataj,self.kernel)
                    b2 = b2 - labelj*(alphaj-alpha_old.item(j-1))*kernel(dataj,dataj,self.kernel)

                    if alphai > 0 and alphai < self.C:
                        self.bias = b1
                    elif alphaj > 0 and alphaj < self.C:
                        self.bias = b2
                    else:
                        self.bias = (b1+b2)/2

                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0
            print "Pass ", passes

    def predict(self, tdata):
        """
        This function predicts the label of the associated datapoints.

        :param tdata: data matrix :math:`D\in\mathbb{R}^{d \\times m}`, where
                     :math:`d` is the dimensionality and
                     :math:`m` is the number of testing samples
        :type tdata: numpy array
        :return: projected test data
        :rtype:  numpy array
        """
        test_dim = tdata.shape[0]

        if self.dim != test_dim:
          raise Exception("ERROR: dimensionality of test data must be consistent.")

        ntest = tdata.shape[1]
        f_evals = self.evaluate(tdata)
        predicted_labels = zeros((1,ntest))

        # find associated indices
        inds1 = where(f_evals >= 0)
        inds2 = where(f_evals < 0)
        predicted_labels[:,inds1] = 1
        predicted_labels[:,inds2] = -1

	return predicted_labels

    def evaluate(self, tdata):
        """
        This function computes the output of the kernel SVM using the kernel, current weight vector and bias.

        :param tdata: data matrix :math:`D\in\mathbb{R}^{d \\times m}`, where
                     :math:`d` is the dimensionality and
                     :math:`m` is the number of testing samples
        :type tdata: numpy array
        :return: projected test data
        :rtype:  numpy array
        """
        test_dim = tdata.shape[0]

        if self.dim != test_dim:
          raise Exception("ERROR: dimensionality of test data must be consistent.")

	K_test = kernel(self.svecs, tdata, self.kernel)
        weights = self.alpha*self.slabels

	f_eval = dot(weights, K_test)
	return f_eval

    def evaluateboundary(self, data, gridsizeX, gridsizeY):
        """
        This function plots the SVM decision boundary for the 2D case with respect to the data.

        :param data: data matrix :math:`D\in\mathbb{R}^{d \\times n}`, where
                     :math:`d` is the dimensionality and
                     :math:`n` is the number of testing samples
        :type data: numpy array
        """
        # parse arguments
        if gridsizeX <= 1 or gridsizeY <= 1:
          raise Exception("ERROR: illegal gridsize. Input integers.")

        # check to see if 2D
        dim = data.shape[0]

        if dim != 2:
          raise Exception("ERROR: can only plot 2D data.")

        if self.dim != dim:
          raise Exception("ERROR: dimensionality of test data must be consistent.")

        # compute limits
        xmin = amin(data[0,:])
        xmax = amax(data[0,:])
        ymin = amin(data[1,:])
        ymax = amax(data[1,:])

        # create grid
        x = linspace(xmin,xmax,num=gridsizeX)
        y = linspace(ymin,ymax,num=gridsizeY)

        xv, yv = meshgrid(x, y)
        vals = zeros((gridsizeX,gridsizeY))

        # evaluate
        for i in xrange(0,gridsizeX-1):
          xval = xv[:,i:i+1]
          yval = yv[:,i:i+1]
          eval_data = hstack((xval,yval))
          fval = self.evaluate(eval_data.T)
          vals[:,i:i+1] = fval.T

	return xv, yv, vals


