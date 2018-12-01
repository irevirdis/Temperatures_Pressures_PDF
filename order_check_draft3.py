import numpy as np
from equadratures import *

#----------------------------------------------------------------#
#  Part 1: Polyint class instance.
#----------------------------------------------------------------#

myBasis   = Basis('Tensor grid')
uni_1  = Parameter(order=5, distribution='uniform', lower=-10, upper =10)
uni_2  = Parameter(order=5, distribution='uniform', lower=-10, upper =10)
myPolyint = Polyint([uni_1, uni_2, uni_1, uni_2], myBasis)

#----------------------------------------------------------------#
#   Part 2:    
#   calculation of efficiency given the information 
#   of temperatures and pressures correlation.
#----------------------------------------------------------------#
coeff = list()

def repeat_for_each_set_of_pnts():
    for k in range(4):
       def fnc_def_pnts():
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
          
           #----------------------------------------------------------------#
           #   calculation of efficiency using correlated points
           corr_pnts = nataf_obj.getCorrelatedSamples(N=3000)
           quad_pnts = nataf_obj.U2C(quad)
          
           #----------------------------------------------------------------#
           #   calculation of efficiency using uncorrelated points
           uncorr_pnts = nataf_obj.C2U(corr_pnts)
           uncorr_quad = nataf_obj.C2U(quad_pnts)
          
           def associations(array):
               eta   = np.zeros(len(array))
               gamma = 1.4
               for i in range(len(array)):
                   t1  = array[i,0]
                   t2  = array[i,1]
                   p1  = array[i,2]
                   p2  = array[i,3]
                   eta[i] = (t1 -t2)/(t1* (1- (p2/p1)**((gamma-1.)/gamma)))
               return eta
          
           eta_list = list()
           eta_list.append(corr_pnts)
           eta_list.append(corr_quad)
           eta_list.append(uncorr_pnts)
           eta_list.append(uncorr_quad)
           eta = list()
          
           def calc_efficiency(lists):   
               for i in range(len(lists)):
                   print 'actual set of points:',lists[i]
                   eta = associations(lists[i])      
                   return eta
           
           return calc_efficiency(eta_list[k])

       coeff.append( myPolyint.computeCoefficients(fnc_def_pnts))
       print 'the coefficients for correlated points are:', coeff[k]

res = repeat_for_each_set_of_pnts()

