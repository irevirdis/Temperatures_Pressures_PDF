import numpy as np
from equadratures import *
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
# -----------------------------------------------------------------------------------------------------------
# Nataf application
#---------------------------------------------------------------------------------------------------------------------
def apply_nataf(R):
    obj = Nataf([M[0], M[1], M[2], M[3]], R)
                                                                  
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

    meanEQ = myStats.mean*100.
    meanMC = np.mean(results_mc)*100.
    
    return meanMC, meanEQ 

def plot2d(R_array, eta_arrayMC, eta_arrayEQ, param):
     plt.figure()
     plt.grid(linewidth=0.5, color='k')
     plt.plot(R_array, eta_arrayMC, 'ro-', label='MonteCarlo')
     plt.plot(R_array, eta_arrayEQ, 'bo-', label='EffectQuad')
     plt.legend(loc='upper right')
     plt.title(param)
     plt.xlabel(r'$\rho$ correlation coefficient')
     plt.ylabel(r'$\eta$ turbine efficiency')
     plt.xticks(R_array)
     plt.show()

def plot3d(R_array_X, R_array_Y, eta_arrayMC, eta_arrayEQ, param):
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')    
    n = len(R_array_X)
    ax.scatter(R_array_X, R_array_Y, eta_arrayMC, c='r', marker='o')
    ax.scatter(R_array_X, R_array_Y, eta_arrayEQ, c='b', marker='^')
    ax.set_xlabel(r'$\rho$ coeff corr temp')
    ax.set_ylabel(r'$\rho$ coeff corr pres')
    ax.set_zlabel(r'$\eta$ efficiency')
    plt.show()
    """
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot_trisurf(R_array_X, R_array_Y, eta_arrayMC, cmap='viridis', edgecolor='b')
    ax.plot_trisurf(R_array_X, R_array_Y, eta_arrayEQ, cmap='magma', edgecolor='r')
    ax.set_xlabel(r'$\rho$ temperatures')
    ax.set_ylabel(r'$\rho$ pressures')
    ax.set_zlabel(r'$\eta$ turbine efficiency')
    plt.show()

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

M = list()

M.append(Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= T01, shape_parameter_B=1.0, upper = T01*1.1, lower = T01*0.95))
M.append(Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= T02, shape_parameter_B=1.0, upper = T02*1.1, lower = T02*0.95))
M.append(Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= P01, shape_parameter_B=1.0, lower = P01*0.90, upper = P01*1.1))
M.append(Parameter(order=5, distribution='truncated-gaussian', shape_parameter_A= P02, shape_parameter_B=1.0, lower = P02*0.90, upper = P02*1.1))

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

R = np.identity(len(M))
R_array = [i*.1 for i in range(10)]
eta_arrayMC = np.zeros(len(R_array))
eta_arrayEQ = np.zeros(len(R_array))

""" Test #1 : Correlation between temperatures but not for pressures
"""
cases_of_correlation = ('temperatures', 'pressures', 'both')
correlation_matrix = list()

for j in range(len(cases_of_correlation)):
    
    if cases_of_correlation[j] == 'temperatures':
    
        for i in range (10):
            R[0,1] = i*.1
            R[1,0] = R[0,1]
           
            res = apply_nataf(R)
                                                                              
            eta_arrayMC[i] = res[0]
            eta_arrayEQ[i] = res[1]
   
        plot2d(R_array, eta_arrayMC, eta_arrayEQ, cases_of_correlation[j])


    if cases_of_correlation[j] == 'pressures':   
        R = np.identity(len(M))
        for i in range(10):
            R[2,3] = i*.1
            R[3,2] = R[2,3]
           
            res = apply_nataf(R)
                                                                              
            eta_arrayMC[i] = res[0]
            eta_arrayEQ[i] = res[1]
   
        plot2d(R_array, eta_arrayMC, eta_arrayEQ, cases_of_correlation[j])

    
    if cases_of_correlation[j] == 'both':
        number_of_rho_values = 10
        square_value = int(number_of_rho_values**2)

        eta_arrayMC_var = np.zeros((square_value))
        eta_arrayEQ_var = np.zeros((square_value))
        R_array_X = np.zeros(square_value) 
        R_array_Y = np.zeros(square_value)
        for i in range(number_of_rho_values):
            for k in range(number_of_rho_values):
                R[0,1] = i*.1
                R[1,0] = R[0,1]
                R[2,3] = k*.1
                R[3,2] = R[2,3]
                               
                res = apply_nataf(R)
                #print 'i:',i, 'k:',k
                index = int(i*number_of_rho_values +k)
                eta_arrayMC_var[index] = res[0]
                eta_arrayEQ_var[index] = res[1]
                R_array_X[index] = (i*.1)
                R_array_Y[index] = (k*.1)
                #print 'index now:' , index

        #print 'R_array_X:'
        #print R_array_X
        #print 'R_array_Y'
        #print R_array_Y
        
        plot3d(R_array_X, R_array_Y, eta_arrayMC_var, eta_arrayEQ_var, cases_of_correlation[j])
    
    

 
