import numpy as np
from equadratures import *

points1 = np.matrix([0.98, 1.51 ,10.2, 2.5*10**5])
points2 = np.matrix([0.99, 1.52 ,10.1, 2.5001*10**5])
points3 = np.matrix([1.0, 1.50 ,10.0, 2.4999*10**5])

tr1 = Parameter(distribution = 'truncated-gaussian', order=5, lower=points1[0,0]*(0.9), upper=1.1*points1[0,0], shape_parameter_A=0.99, shape_parameter_B=1.)
tr4 = Parameter(distribution = 'truncated-gaussian', order=5, lower=points1[0,1]*.99, upper=1.1*points1[0,1],shape_parameter_A = 1.5, shape_parameter_B= 1.)
tr2 = Parameter(distribution = 'truncated-gaussian', order=5, lower=points1[0,2]*(0.9), upper=1.1*points1[0,2], shape_parameter_A = 10.1, shape_parameter_B =1.)
tr3 = Parameter(distribution = 'truncated-gaussian', order=5, lower=points1[0,3]*(0.9), upper=1.1*points1[0,3], shape_parameter_A = 2.5*10**5, shape_parameter_B = 1.)

R = np.matrix([[1., 0.64, 0.4, 0.8],
              [0.64, 1., 0.7, 0.5],
              [0.4, 0.7, 1., 0.2],
              [0.8, 0.5, 0.2, 1.]])

my_Nataf = Nataf([tr1, tr4 , tr2, tr3], R)

p1 = my_Nataf.U2C(points1)
print 'first results:',p1
p2 = my_Nataf.U2C(points2)
print 'second res:', p2
p3 = my_Nataf.U2C(points3)
print 'last:', p3
