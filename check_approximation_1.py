import numpy as np
from equadratures import *

#------------------------------------------------------------------#
# PART 2
# Polyint class instance

myBasis = Basis('Tensor grid')
uni_1   = Parameter(distribution='uniform', order=5, upper=810, lower=790)
uni_2   = Parameter(distribution='uniform', order=5, upper=710, lower=690)
uni_3   = Parameter(distribution='uniform', order=5, upper=9*10**5, lower=11*10**5)
uni_4   = Parameter(distribution='uniform', order=5, upper=4*10**5, lower=6*10**5)
myPolyint = Polyint([uni_1, uni_2, uni_3, uni_4], myBasis)

def efficiency(x):
    t1 = x[0]
    t2 = x[1]
    p1 = x[2]
    p2 = x[3]
    gamma = 1.4
    eta   = (t1 - t2)/(t1 *(1- (p2/p1)**((gamma-1)/gamma)))
    return eta

myPolyint.computeCoefficients(efficiency)
coeff = myPolyint.coefficients

#------------------------------------------------------------------#
#   PART 2
#   generate a polynomial approximation of the efficiency blackbox

max_order = (uni_1.order +1)
#print 'the value of the maximum order is:', max_ordeur
order_info = list()

def poly_approx(t1, t2, p1, p2):
    """ blackbox written to assign the polynomial coeffiecients to the power of 
        temperatures and pressures.
    """
    t = 0 # index of the i_th element inside the array myPolyint.coefficients and Eta array
    for i in range (int(max_order)):
        # i will be the index associated to the power of temperature T1
        for j in range(int(max_order)):
            # j will be associated to the power of temperature T2
            for k in range(int(max_order)):
                # k will be associated to th power of pressure P1
                for h in range(int(max_order)):
                    # h will the power of pressure P2
                   
                    # in the following lines:
                    # x[0]=T1, x[1]=T2, x[2]=P1, x[3]=P2
                    # only the coefficients > 10**3 will be taken into account

                    if coeff[t] > 10**(-3): 
                        #print 'the size of x[0] is:', t1.shape
                        #print 'the type of x[0] is :', type(t1)
                        np.set_printoptions(threshold=np.nan)
                        #print 'the total array of x[0] is:', x1
                        eta =  coeff[t]*(t1**i + t2**j + p1**k + p2**h)
                        #print 'the value of t is:', t
                        sum_of_power     = int(i+j+k+h)
                        order_info.append(sum_of_power) 
                    else:
                        eta = 0.0
                    t = int(t+1) # pass to the following element inside coefficients array
    return eta
#-----------------------------------------------------------------#
#   PART 2
#   Sample with uncorrelated points
#   1) generation of quadrature points

normal = Parameter(distribution='gaussian', shape_parameter_A=0., shape_parameter_B=1., order=5)
stdPoly = Polyint([normal, normal, normal, normal], myBasis)
points = stdPoly.quadraturePoints

myStats = myPolyint.getStatistics()


#--------------- statistics of uncorrelated points ---------------#

res = np.zeros((len(points),1))
for i in range(len(points)):
    res[i] = poly_approx(points[i,0], points[i,1], points[i,2], points[i,3])
myPolyint.computeCoefficients(res)
print '---------------------------'
print 'Uncorrelated data:'
print 'mean:', myStats.mean
print 'variance:', myStats.variance
print 'the maximum value of order is:', max(order_info)
print 'the number of terms inside the polynomial approximantion is:', len(order_info)
print '---------------------------'

# ------------- statistics for uncorrelated points ---------------#

# 2) generation of all the elements due to the Nataf instance
# 2.1 truncated gaussian distributions, centered with respect to the mean values of temperatures and pressures
T01 = 800.
distr1 = Parameter(distribution='truncated-gaussian', shape_parameter_A = T01, shape_parameter_B = 1., order=5, upper=T01*1.1, lower=T01*.9)
T02 = 700.
distr2 = Parameter(distribution='truncated-gaussian', shape_parameter_A = T02, shape_parameter_B = 1., order=5, upper=T02*1.1, lower=T02*.9)
P01 = 10*10**5
distr3 = Parameter(distribution='truncated-gaussian', shape_parameter_A = P01, shape_parameter_B = 1., order=5, upper=P01*1.1, lower=P01*.9)
P02 = .5*10**5
distr4 = Parameter(distribution='truncated-gaussian', shape_parameter_A = P02, shape_parameter_B = 1., order=5, upper=P02*1.1, lower=P02*.9)

# 2.2 Correlation matrix
R = np.array([[1., .3, 0., 0.],
              [.3, 1., 0., 0.],
              [0., 0., 1., .6],
              [0., 0., .6, 1.]])
# 2.3 Instance to Nataf class
myNataf = Nataf([distr1, distr2, distr3, distr3], R)

# 2.4 Correlation among quadrature points
pc  = myNataf.U2C(points)
res = np.zeros((len(pc),1))

# 2.5 application of the polynomial approximation to the correlated points
for i in range(len(pc)):
    res[i] = poly_approx(pc[i,0], pc[i,1], pc[i,2], pc[i,3])
myPolyint.computeCoefficients(res)
np.set_printoptions(threshold=np.nan)


# 2.6 print of mean and variance
print '---------------------------'
print 'Correlated data:'
print 'mean:', myStats.mean
print 'variance:', myStats.variance
print 'the maximum value of order info is:', max(order_info)
print 'the number of terms inside the polynomial approximantion is:', len(order_info)
print '---------------------------'






