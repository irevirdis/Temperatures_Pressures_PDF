import numpy as np
from equadratures import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

""" Mean values for temperatures and variances;
    gamma: adiabatic isentropic exponent for gas
"""
T01 = 800.
T02 = 700.
P01 = 10*10**5
P02 = 5*10**5
gamma = 1.4

""" construction of the correlated points: 
    temperatures and pressures
"""

t1 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = T01, shape_parameter_B = 1., upper = T01*1.1, lower = T01*.9)
t2 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = T02, shape_parameter_B = 1., upper = T02*1.1, lower = T02*.9)
p1 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = P01, shape_parameter_B = 1., upper = P01*1.1, lower = P01*.9)
p2 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = P02, shape_parameter_B = 1., upper = P02*1.1, lower = P02*.9)

""" quadrature rule
"""
myBasis = Basis('Tensor grid')
normal = Parameter(order=5, distribution='gaussian', shape_parameter_A = 0., shape_parameter_B = 1.)
standardPoly = Polyint([normal, normal, normal, normal], myBasis)
quad = standardPoly.quadraturePoints

# definition of the correlation matrix
R = np.matrix([[1., .3, 0., 0.],
               [.3, 1., 0., 0.],
               [0., 0., 1., .6],
               [0., 0., .6, 1.]])

# definition of a nataf object
nataf_obj = Nataf([t1, t2, p1, p2], R) 

# correlated samples
corr_pnts = nataf_obj.getCorrelatedSamples(N=3000)
quad_pnts = nataf_obj.U2C(quad)

#plt.figure()
#plt.grid()
#plt.plot(quad_pnts[:,2], quad_pnts[:,3], 'bo')
#plt.title('quadrature points')
#plt.show()

def efficiency(t1, t2, p1, p2, gamma):
    eta = (t1 -t2)/(t1* (1- (p2/p1)**((gamma-1.)/gamma)))
    return eta

efficiency_corr = np.zeros((len(corr_pnts),1))
efficiency_quad = np.zeros((len(quad_pnts),1))
x = np.zeros((len(efficiency_corr), 1))
y = np.zeros((len(efficiency_corr), 1))
xq = np.zeros((len(efficiency_quad), 1)) 
yq = np.zeros((len(efficiency_quad), 1)) 
for i in range(len(x)):
    x[i] = corr_pnts[i,0]
    y[i] = corr_pnts[i,1]
    efficiency_corr[i] = efficiency(corr_pnts[i,0], corr_pnts[i,1], corr_pnts[i,2], corr_pnts[i,3], gamma)

for i in range(len(quad_pnts)):
    xq[i] = quad_pnts[i,0]
    yq[i] = quad_pnts[i,1]
    efficiency_quad[i] = efficiency(quad_pnts[i,0], quad_pnts[i,1], quad_pnts[i,2], quad_pnts[i,3], gamma)
#    print (quad_pnts[i,0], quad_pnts[i,1], quad_pnts[i,2], quad_pnts[i,3], efficiency_quad[i])

#plt.figure()
#plt.grid()
#plt.plot(xq, yq, 'bo')
#plt.title('quadrature points')
#plt.show()

#print 'xq is:'
#print xq
#print 'x has lenght:', (x.shape)
#print 'y has lenght:', (y.shape)
#print 'z has lenght:', efficiency_corr.shape

""" PCOLOR FOR CONTOUR PLOT OF EFFICIENCY :
"""
plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.scatter(x,y, c=efficiency_corr, s=50)
plt.gray()
plt.colorbar()
plt.xlabel('Temperature 1')
plt.ylabel('Temperature 2')
plt.title('MC sampling')
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.scatter(xq,yq, c=efficiency_quad, s=100)
plt.gray()
plt.colorbar()
plt.xlabel('Temperature 1')
plt.ylabel('Temperature 2')
plt.title('EQ sampling')
plt.show()

"""     PART 2: MAP INTO A STANDARD SPACE
"""

uncorr_pnts = nataf_obj.C2U(corr_pnts)
uncorr_quad = nataf_obj.C2U(quad_pnts)

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(uncorr_pnts[:,0], uncorr_pnts[:,1], 'bo')
plt.title('std space for MQ')
plt.xlabel('Temperature 1')
plt.ylabel('Temperature 2')
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.plot(uncorr_quad[:,0], uncorr_quad[:,1], 'bo')
plt.title('std space for EQ')
plt.xlabel('Temperature 1')
plt.ylabel('Temperature 2')
plt.show()

x = np.zeros((len(uncorr_pnts),1))
y = np.zeros((len(x),1))
z = np.zeros((len(x),1))
#plt.figure()
#plt.grid()
#plt.plot(x, y, 'bo')
#plt.title('std space')
#plt.show()
""" in the following lines the mean values of temperatures and
    pressures have been added to std pnts because a correct
    value of efficiency can be obtained.
"""
for i in range(len(x)):
    x[i] = uncorr_pnts[i,0] + T01
    y[i] = uncorr_pnts[i,1] + T02
    p1 = uncorr_pnts[i,2] + P01
    p2 = uncorr_pnts[i,3] + P02
    z[i] = efficiency(x[i], y[i], p1, p2, gamma)

xq = np.zeros((len(uncorr_quad),1))
yq = np.zeros((len(xq),1))
zq = np.zeros((len(xq),1))
#plt.figure()
#plt.grid()
#plt.plot(xq, yq, 'bo')
#plt.title('std space')
#plt.show()
for i in range(len(xq)):
    xq[i] = uncorr_quad[i,0] + T01
    yq[i] = uncorr_quad[i,1] + T02
    p1 = uncorr_quad[i,2] + P01
    p2 = uncorr_quad[i,3] + P02
    zq[i] = efficiency(xq[i], yq[i], p1, p2, gamma)

""" PCOLOR FOR CONTOUR PLOT OF EFFICIENCY :
"""
plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.scatter(x,y, c=z, s=50 )
plt.gray()
plt.colorbar()
plt.title('MC sampling')
plt.xlabel('Temperature 1')
plt.ylabel('Temperature 2')
plt.show()

plt.figure()
plt.grid(linewidth=0.5, color='k')
plt.scatter(xq,yq, c=zq, s=100)
plt.gray()
plt.colorbar()
plt.title('EQ sampling')
plt.xlabel('Temperature 1')
plt.ylabel('Temperature 2')
plt.show()

