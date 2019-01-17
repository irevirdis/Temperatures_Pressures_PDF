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

