import numpy as np
from equadratures import *
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# --------------------------------------------------------------------------------------------------------------------
#
# EFFICIENCY DEFINITION
#
#---------------------------------------------------------------------------------------------------------------------
def blackbox(t1, t2, p1, p2):
    """ Efficiency definition:
        numerator and denominator
        have been devided by Cp,
        assumed as a constant.
    """
    eta       = (t1 - t2) / (t1 * (1 - (p2/p1)**(0.4/1.4)))
    return eta

# --------------------------------------------------------------------------------------------------------------------
#
# INPUTS
#
#---------------------------------------------------------------------------------------------------------------------
gamma = 1.4
T01   = 800.
T02   = 700.
P01   = 10*10**5
P02   = 5*10**5
t1 = Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= T01, shape_parameter_B=1.0, upper = T01*1.1, lower = T01*0.95)
t2 = Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= T02, shape_parameter_B=1.0, upper = T02*1.1, lower = T02*0.95)
p1 = Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= P01, shape_parameter_B=1.0, lower = P01*0.90, upper = P01*1.1)
p2 = Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= P02, shape_parameter_B=1.0, lower = P02*0.90, upper = P02*1.1)


# --------------------------------------------------------------------------------------------------------------------
#
# POLYNOMIAL CONSTRUCTION
#
#---------------------------------------------------------------------------------------------------------------------
# Construct a polynomial in the uncorrelated normal space
myBasis = Basis('Tensor grid')
normal = Parameter(order=5, distribution='gaussian', shape_parameter_A=0.0, shape_parameter_B=1.0)
standardPoly = Polyint([normal, normal, normal, normal], myBasis)
p = standardPoly.quadraturePoints


# Correlation matrix -- note we only correlate T01 with T02, and P01 with P02.
R = np.matrix([[1.0, 0.3, 0.0, 0.0],
               [0.3, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.6],
               [0.0, 0.0, 0.6, 1.0]])

#R = np.matrix([[1.0, 0.8, 0.76, 0.76],
#               [0.8, 1.0, 0.76, 0.76],
#               [0.76, 0.76, 1.0, 0.75],
#               [0.76, 0.76, 0.75, 1.0]])

#R = np.matrix([[1.0, 0.0, 0.0, 0.0],
#               [0.0, 1.0, 0.0, 0.0],
#               [0.0, 0.0, 1.0, 0.0],
#               [0.0, 0.0, 0.0, 1.0]])



# Nataf transformation
obj = Nataf([t1, t2, p1, p2], R)


pc = obj.U2C(p)
mc_pts = obj.getCorrelatedSamples(N=3000)

results = blackbox(pc[:,0], pc[:,1], pc[:,2], pc[:,3])
results_mc = blackbox(mc_pts[:,0], mc_pts[:,1], mc_pts[:,2], mc_pts[:,3])
standardPoly.computeCoefficients(results)
myStats = standardPoly.getStatistics()

print '==============EQ==============='
print 'mean', myStats.mean * 100
print '2-sigma', np.sqrt(myStats.variance) * 1.96

print '==============MC==============='
print 'mean', np.mean(results_mc)* 100
print '2-sigma', np.sqrt(np.var(results_mc)) * 1.96

