"""
Abstract base class for kernel machine objects. Indicates 
methods that must always be completed. 
"""
import abc

class KMach(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def process(self, input):
        """Take data as input, perform processing that generates kernel 
           machine."""
        return

