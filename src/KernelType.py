from numpy import shape
# import scipy
class KernelType:
    """
    Implements a class which initializes the kernel type and parameters. 
    Currently supported kernels are

    * polynomial - :math:`k(x,y):= \\langle x,y \\rangle^{\gamma} + c`,
      where :math:`\ x,y,\in\mathbb{R}^d, \gamma \in\mathbb{N}, c\in\mathbb{R}`.    
    * sigmoid - :math:`k(x,y):= \\tanh(\\alpha x^Ty + c)`, 
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\alpha, c\in\mathbb{R}`. 
    * gaussian - :math:`k(x,y):= \exp{\left(\\frac{\|x-y\|^2}{2\sigma^2}\\right)}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\sigma\in\mathbb{R}_+`. 
    * laplacian - :math:`k(x,y):= \exp{\left(\\frac{\|x-y\|}{\sigma}\\right)}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\sigma\in\mathbb{R}_+`. 
    * cauchy - :math:`k(x,y):= \\frac{1}{1 + \\frac{\|x-y\|^2}{\sigma^2}}`,
      where :math:`\ x,y,\in\mathbb{R}^d`, and :math:`\\sigma\in\mathbb{R}_+`. 

    Example usage:: 

        k_name = "sigmoid"
        k_params = array( [0.5, 1.2] ) # numpy array
        k = KernelType(k_name,k_params)

    :param k_name: Kernel name
    :type k_name: string
    :param k_params: Kernel parameters
    :type k_params: numpy mat 
    :return: `KernelType object`
    :rtype:  KernelType
    """
    def __init__(self, k_name, k_params):
        # need to check if parameters passed are legitimate
        if k_name == "polynomial": 
            k_size = k_params.shape
            if k_size[0] == 2:
                self.name   = k_name
                self.params = k_params 
            else: 
                raise Exception("Incorrect number of parameters: polynomial kernel needs degree and bias")
        elif k_name == "gaussian":
            k_size = k_params.shape
            if k_size[0] == 1:
                self.name   = k_name
                self.params = k_params 
            else: 
                raise Exception("Incorrect number of parameters: gaussian kernel needs bandwidth")
        elif k_name == "sigmoid": 
            k_size = k_params.shape
            if k_size[0] == 2:
                self.name   = k_name
                self.params = k_params 
            else: 
                raise Exception("Incorrect number of parameters: sigmoid kernel needs bandwidth")
        elif k_name == "laplacian":
            k_size = k_params.shape
            if k_size[0] == 1:
                self.name   = k_name
                self.params = k_params 
            else: 
                raise Exception("Incorrect number of parameters: laplacian kernel needs bandwidth")
        elif k_name == "cauchy":
            k_size = k_params.shape
            if k_size[0] == 1:
                self.name   = k_name
                self.params = k_params 
            else: 
                raise Exception("Incorrect number of parameters: cauchy kernel needs bandwidth")
        else:
            raise Exception("Invalid kernel type: supported types are polynomial, gaussian, laplacian, cauchy, sigmoid\n")

