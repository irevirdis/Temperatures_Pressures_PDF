"""The Truncated Gaussian distribution."""
import numpy as np
from scipy.special import erf, erfinv, gamma, beta, betainc, gammainc
from distribution import Distribution
from scipy.stats import truncnorm
from gaussian import *

class TruncatedGaussian(Distribution):
    """
    The class defines a Truncated-Gaussian object. It is the child of Distribution.
    :param double mean:
		Mean of the truncated Gaussian distribution.
	:param double variance:
		Variance of the truncated Gaussian distribution.
    :param double lower:
        Lower bound of the truncated Gaussian distribution.
    :param double upper:
        Upper bound of the truncated Gaussian distribution.
    """
    def __init__(self, mean, variance, lower, upper):
        if (mean is not None) and (variance is not None) and (lower is not None) and (upper is not None):
            self.meanParent = mean
            self.varianceParent = variance
            self.std    = Gaussian(mean=0.0, variance=1.0)
            self.parent = Gaussian(mean= self.meanParent, variance=self.varianceParent)
            self.lower = lower 
            self.upper = upper
            self.skewness = 0.0
            self.kurtosis = 0.0
            self.bounds = np.array([-np.inf, np.inf]) 
            self.beta  = (self.upper - self.meanParent)/self.varianceParent  
            self.alpha = (self.lower - self.meanParent)/self.varianceParent
            num = self.std.getPDF(points=self.beta)-self.std.getPDF(points=self.alpha)
            den = self.std.getCDF(points=self.beta)-self.std.getCDF(points=self.alpha)
            self.mean = self.meanParent - self.varianceParent*(num/den)
            #print 'da costruttore: mean', self.mean
            
            num_i = self.beta*self.std.getPDF(points=self.beta)-self.alpha*self.std.getPDF(points=self.alpha)
            den   = self.std.getCDF(points=self.beta)-self.std.getCDF(points=self.alpha)
            num_ii= self.std.getPDF(points=self.beta)-self.std.getPDF(points=self.alpha)
            self.variance = self.varianceParent*(1-(num_i/den)-(num_ii/den)**2)
            #print 'da costruttore: variance:', self.variance
            self.sigma = np.sqrt(self.variance)

    def getDescription(self):
        """
        A description of the truncated Gaussian.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
        :return:
            A string describing the truncated Gaussian.
        """
        text = "A truncated Gaussian distribution with a mean of "+str(self.mean)+" and a variance of "+str(self.variance)+", and a lower bound of "+str(self.lower)+" and an upper bound of "+str(self.upper)+"."
        return text

    def getPDF(self, points=None):
        """
        A truncated Gaussian probability distribution.

        :param truncated Gaussian self:
            An instance of the truncated Gaussian class.
		:param integer N:
            Number of equidistant points over the support of the distribution; default value is 500.
        :return:
            An array of N equidistant values over the support of the distribution.
        :return:
            Probability density values along the support of the truncated Gaussian distribution.
        """
        w = np.zeros(len(points))
        if points is not None:
            for i in range(len(points)):
                if points[i] <= self.lower:
                    w[i] = 0
                elif self.upper <= points[i]:
                    w[i] = 0
                else:
                    num = self.parent.getPDF(points = points[i])
                    den = self.parent.getCDF(points = self.upper)-self.parent.getCDF(points =self.lower)
                    w[i] = num/den
            return w
        else: 
            raise(ValueError, 'Please digit the points that have to be evaluated by getPDF method.')

    def getCDF(self, points = None):
        """
        A truncated Gaussian cumulative density function.

	    :param truncated Gaussian self:
            An instance of the Gaussian class.
        :param integer N:
            Number of points for defining the cumulative density function; default value is 500.
        :return:
            An array of N equidistant values over the support of the truncated Gaussian.
        :return:
            Gaussian cumulative density values.
        """
        if points is not None:
            num = self.parent.getCDF(points=points) - self.parent.getCDF(points=self.lower)
            den = self.parent.getCDF(points=self.upper) - self.parent.getCDF(points=self.lower)
            w = num/den
            return w
        else:
            raise(ValueError, 'Please digit the points that have to be evaluated by getPDF method.')

    def getiCDF(self, xx):
        """
        A truncated Gaussian inverse cumulative density function.
                                                                                               
            :param truncated Gaussian self:
            An instance of the Gaussian class.
        :param array xx:
            Array of points in which will be evaluated the inverse cumulative density function
        :return:
            Gaussian inverse cumulative density values.
        """
        #obj= truncnorm(a = self.alpha, b = self.beta,loc=self.meanParent, scale = self.varianceParent)
        #return obj.ppf(xx)
        
        num = self.parent.getCDF(points=xx) - self.parent.getCDF(points=self.lower)
        den = self.parent.getCDF(points=self.upper) - self.parent.getCDF(points=self.lower)
        p = num + self.parent.getCDF(points=self.lower)
        w = self.parent.getiCDF(p) 
        return w

#     def getSamples(self, m=None):
#        """ Generates samples from the Truncated-Gaussian distribution.
#
#        :param trunc-norm self:
#             An instance of the Truncated-Gaussian class.
#        :param integer m:
#             Number of random samples. If no value is provided, a default of     5e5 is assumed.
#        :return:
#             A N-by-1 vector that contains the samples.
#        """
#        obj= truncnorm(a = self.alpha, b = self.beta, loc=self.meanParent, scale=self.varianceParent)
#        if m is not None:
#            number = m
#        else:
#            number = 500000
#        return obj.rvs(size=number)

    
