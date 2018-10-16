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

""" PART 2: PLOT OF THE PDFs : the problem should be into the 3rd and the 4th
    marginal.
    HYPOTHESIS: THE CORRELATION AMONG DATA DOES NOT CHANGE THE MARGINALS TYPE.
"""
pdf_pc_t1 = t1.getPDF(pc[:,0])
print 'from testing script: the pdf of correlated t1 is:'
print pdf_pc_t1
pdf_pc_t2 = t2.getPDF(pc[:,1])
pdf_pc_p1 = p1.getPDF(pc[:,2])
print 'from testinf script: the pdf of correlated p1 is:'
print pdf_pc_p1
pdf_pc_p2 = p2.getPDF(pc[:,3])

pdf_mc_t1 = t1.getPDF(mc_pts[:,0])
pdf_mc_t2 = t2.getPDF(mc_pts[:,1])
pdf_mc_p1 = p1.getPDF(mc_pts[:,2])
pdf_mc_p2 = p2.getPDF(mc_pts[:,3])

axes = plt.gca()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(mc_pts[:,0], pdf_mc_t1, 'rx', label='t1_mc')
plt.plot(pc[:,0], pdf_pc_t1, 'ko', label='t1_pc')
plt.legend(loc='upper left')
plt.title('temperatures')
#plt.axis('equal')
axes.set_xlim(T01*0.95, T01*1.1)
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
#plt.plot(pc[:,0], pdf_pc_t1, 'bo', label='t1_pc')
plt.plot(mc_pts[:,1], pdf_mc_t2, 'rx', label='t2_mc')
plt.plot(pc[:,1], pdf_pc_t2, 'ko', label='t2_pc')
#plt.plot(mc_pts[:,0], pdf_mc_t1, 'go', label='t1_mc')
plt.legend(loc='upper left')
plt.title('temperatures')
#plt.axis('equal')
axes.set_xlim(T02*0.95, T02*1.1)
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(mc_pts[:,2], pdf_mc_p1, 'ro', label='p1_mc')
plt.plot(pc[:,2], pdf_pc_p1, 'ko', label='p1_pc')
plt.legend(loc='upper left')
#plt.axis('equal')
axes.set_xlim(P01*0.9, P02*1.1)
plt.title('pressure')
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
#plt.plot(pc[:,2], pdf_pc_p1, 'bo', label='p1_pc')
plt.plot(mc_pts[:,3], pdf_mc_p2, 'ro', label='p2_mc')
plt.plot(pc[:,3], pdf_pc_p2, 'ko', label='p2_pc')
#plt.plot(mc_pts[:,2], pdf_mc_p1, 'go', label='p1_mc')
plt.legend(loc='upper left')
axes.set_xlim(P02*0.9, P02*1.1)
#plt.axis('equal')
plt.title('pressure')
plt.show()

""" test of Nataf transformation 
"""
marginal_pc_p1 = pc[:,2]
marginal_pc_p2 = pc[:,3]
marginal_mc_p1 = mc_pts[:,2]
marginal_mc_p2 = mc_pts[:,3]

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(marginal_mc_p1, marginal_mc_p2, 'ro', label='mc')
plt.plot(marginal_pc_p1, marginal_pc_p2, 'bo', label='pc')
plt.legend(loc='upper left')
plt.title('quadrature points VS mc FOR PRESSURES')
plt.show()

marginal_pc_t1 = pc[:,0]
marginal_pc_t2 = pc[:,1]
marginal_mc_t1 = mc_pts[:,0]
marginal_mc_t2 = mc_pts[:,1]

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(marginal_mc_t1, marginal_mc_t2, 'ro', label='mc')
plt.plot(marginal_pc_t1, marginal_pc_t2, 'bo', label='pc')
plt.legend(loc='upper left')
plt.title('quadrature points VS mc FOR TEMPERATURES')
plt.show()

""" 3rd: pdf of uncorrelated.
"""
t1_u = t1.getSamples(1000)
t2_u = t2.getSamples(1000)
p1_u = p1.getSamples(1000)
p2_u = p2.getSamples(1000)

pdf_t1_u = t1.getPDF(t1_u)
pdf_t2_u = t2.getPDF(t2_u)
pdf_p1_u = p1.getPDF(p1_u)
pdf_p2_u = p2.getPDF(p2_u)

#print 'pdf for uncorrelated :'
#print pdf_p1_u

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(p1_u, pdf_p1_u, 'bo', label='p1u')
plt.legend(loc='upper left')
plt.title('p1u')
axes.set_xlim(T01*0.95, T01*1.1)
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(p2_u, pdf_p2_u, 'ro', label='p2u')
plt.legend(loc='upper left')
plt.title('p2u')
axes.set_xlim(T02*0.95, T02*1.1)
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(t1_u, pdf_t1_u, 'bo', label='t1u')
plt.legend(loc='upper left')
axes.set_xlim(P01*0.9, P02*1.1)
plt.title('t1u')
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(t2_u, pdf_t2_u, 'ro', label='t2u')
plt.legend(loc='upper left')
axes.set_xlim(P02*0.9, P02*1.1)
plt.title('t2u')
plt.show()




