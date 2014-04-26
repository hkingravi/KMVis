
class KMVisError(Exception):
    """Generic exception thrown specific to KMVis."""
    pass

class KernelParametersError(KMVisError):
    """Exception thrown when kernel parameters are incorrect."""
    pass

class KernelTypeError(KMVisError):
    """Exception thrown when kernel type is incorrect."""
    pass