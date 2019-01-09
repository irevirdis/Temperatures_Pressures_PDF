import numpy as np
from equadratures import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#----------------------------------------------------------------#
#  Polyint class instance.

myBasis   = Basis('Tensor grid')
uni_1  = Parameter(order=5, distribution='uniform', upper=810, lower =790)
uni_2  = Parameter(order=5, distribution='uniform', upper=710, lower =690)
uni_3  = Parameter(order=5, distribution='uniform', upper=9*10**5, lower=11*10**5)
uni_4  = Parameter(order=5, distribution='uniform', upper=4*10**5, lower=6*10**5)
myPolyint = Polyint([uni_1, uni_2, uni_3, uni_4], myBasis)

#def check_order(fnc):
#    coefficients = myPolyint.computeCoefficients(fnc)
#    return coefficients

#----------------------------------------------------------------#
#   part 2:    
#   calculation of efficiency given the information 
#   of temperatures and pressures correlation.

#----------------------------------------------------------------#
def efficiency(x):
    t1 = x[0]
    t2 = x[1]
    p1 = x[2]
    p2 = x[3]
    gamma =1.4

    """ construction of the correlated points: 
        temperatures and pressures
    """
    T01 = 800.
    T02 = 700.
    P01 = 10*10**5
    P02 = 5*10**5

    distr_t1 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = T01, shape_parameter_B = 1., upper = T01*1.1, lower = T01*.9)
    distr_t2 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = T02, shape_parameter_B = 1., upper = T02*1.1, lower = T02*.9)
    distr_p1 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = P01, shape_parameter_B = 1., upper = P01*1.1, lower = P01*.9)
    distr_p2 = Parameter(distribution='truncated-gaussian', order=5, shape_parameter_A = P02, shape_parameter_B = 1., upper = P02*1.1, lower = P02*.9)

    # definition of the correlation matrix
    R = np.matrix([[1., .3, 0., 0.],
                   [.3, 1., 0., 0.],
                   [0., 0., 1., .6],
                   [0., 0., .6, 1.]])
    #------------------------------------------------------------------------------#
    # the lines from 58 to 70 have been commented in the case of uncorrelad 
    # temperatures and pressures
    #------------------------------------------------------------------------------#
    ## definition of a nataf object
    #nataf_obj = Nataf([distr_t1, distr_t2, distr_p1, distr_p2], R) 

    # arrays of samples
    #unco_pnts = np.matrix([t1, t2, p1, p2])
    #corr_pnts = nataf_obj.U2C(unco_pnts) 

    ## definition of the two sets that will be used 
    ## and turbine efficiency

    #t1 = corr_pnts[0,0]
    #t2 = corr_pnts[0,1]
    #p1 = corr_pnts[0,2]
    #p2 = corr_pnts[0,3]
    #------------------------------------------------------------------------------#  
    eta = (t1 -t2)/(t1* (1- (p2/p1)**((gamma-1.)/gamma)))
   
    return eta

myPolyint.computeCoefficients(efficiency)

#--------------------------------------------------------------#
# from EQ documentation: comparison among coefficients
#--------------------------------------------------------------#
x,y,z, max_order = utils.twoDgrid(myPolyint.coefficients, myPolyint.basis.elements)
G = np.log10(np.abs(z))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cax = plt.scatter(x,y,s=480, marker='o', c=G, cmap='jet', alpha=.8)#, vmax=2, vmin=-12)
plt.xlim(-.5, max_order)
plt.ylim(-.5, max_order)
adjust_spines(ax, ['left', 'bottom'])
ax.set_axisbelow(True)
plt.xlabel('coeff 1')
plt.ylabel('coeff 2')
plt.title("Uncorrelated Temp.and Pres.")
cbar = plt.colorbar(extend ='neither', spacing='proportional', orientation='vertical', shrink=.8, format="%.0f")
cbar.ax.tick_params(labelsize=13)
plt.savefig('Coefficients_uncorrelated1.png', dpi=200, bbox_inches='tight')
plt.close()
#print 'the coefficients are:', myPolyint.coefficients
#print 'the size of coefficients are:', (myPolyint.coefficients).shape
