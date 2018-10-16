""" Class for the Indipendent Component Analysis in N-dimensional case,
    the method aims to reveal hidden factors that undelie sets of random
    variables.
    Reference for theory: notes of Dr. Mohsen Naqvi, 'Fundamentals of PCA
    ICA and IVA', UDRC Summer school, 23 July 2015, New Castle University.
""" 
import numpy as np
from parameter import Parameter

class Ica(object):
    """ The class defines an ICA (Independent Component Analysis.
        The restrictions are:
            - The sources are assumed to be independent of each other
            - All but one of the marginals must be non-gaussian

        :param 
        :param
    """
    def __init__(self, D = None):
        if D is None:
            raise(ValueError, 'Distributions must be given')
        else:
            self.D = D



