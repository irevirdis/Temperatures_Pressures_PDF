"""Finding coefficients via least squares."""
from .parameter import Parameter
from .basis import Basis
from .poly import Poly
import numpy as np
from utils import evalfunction, evalgradients, cell2matrix
from scipy.linalg import qr, svd, lu
from qr import solveCLSQ
import matplotlib.pyplot as plt
from convex import maxdet, binary2indices

class Polylsq(Poly):
    """
    This class defines a Polylsq (polynomial via least squares) object.

    :param Parameter parameters: A list of parameters.
    :param Basis basis: A basis selected for the multivariate polynomial.
    :param string mesh: A mesh used for the sampling scheme. Options include:
        'Tensor': A tensor grid based on the order of each parameter;
        'Chebyshev': A chebyshev sampling over the domain;
        'Uniform': A uniform grid over the domain, where the number of points in each direction is based on the order of the parameter;
        'Random': A random sampling over the domain.
    :param string optimization: The optimization strategy used for subsampling points from the underlying mesh. Options include:
        'Greedy-QR': Subselecting points using QR column pivoting.
        'Greedy-SVD': Subselecting points using the SVD followed by QR with column pivoting.
        'Newton': Subselecting points via a determinant maximization heuristic, solved using Newton's method.
    :param double oversampling: The ratio of the number of points to the number of coefficients. It can be any value above 1.0 and below a value based on the total number of samples in the mesh.
    :param string gradients: If gradient values are provided, then select this option to 'True'; default value is 'False'.
    
    Attributes:
        * **self.mesh**: 
        * **self.optimization**: 
        * **self.oversampling**:
    """
    def __init__(self, parameters, basis, mesh, optimization=None, oversampling=None, gradients=False):
        super(Polylsq, self).__init__(parameters, basis)
        self.mesh = mesh
        self.optimization = optimization
        if self.optimization is None:
            self.optimization = 'none'
        if oversampling is None:
            self.oversampling = 1.0
        else:
            self.oversampling = oversampling
        self.gradients = gradients
        n = self.basis.cardinality
        m_refined = int(np.round(self.oversampling * n))
        m_big = 1
        for i in range(0, self.dimensions):
            m_big = (self.parameters[i].order  + 1) * m_big
        if self.mesh.lower() == 'tensor':
            if self.optimization.lower() == 'random':
                pts, wts = [] , [] # empty points and weights to save memory!
            else:
                pts, wts = super(Polylsq, self).getTensorQuadratureRule() # original weights sum up to 1
        elif self.mesh.lower() == 'chebyshev':
            pts = np.cos(np.pi * np.random.rand(m_big, self.dimensions ))
            wts =  1.0/np.sum( super(Polylsq, self).getPolynomial(pts)**2 , 0)
            wts = wts * 1.0/np.sum(wts)
        elif self.mesh.lower() == 'tensorcheck':
            pts, _ = super(Polylsq, self).getTensorQuadratureRule()
            wts =  1.0/np.sum( super(Polylsq, self).getPolynomial(pts)**2 , 0)
            wts = wts * 1.0/np.sum(wts)
        elif self.mesh.lower() == 'uniform':
            pts = np.zeros((m_big, self.dimensions))
            for i in range(0, self.dimensions):
                univariate_samples = np.linspace(self.parameters[i].lower, self.parameters[i].upper, m_big)
                for j in range(0, m_big):
                    pts[j, i] = univariate_samples[j]
            wts =  1.0/np.sum( super(Polylsq, self).getPolynomial(pts)**2 , 0)
            wts = wts * 1.0/np.sum(wts)
        elif self.mesh.lower() == 'random':
            pts = np.zeros((m_big, self.dimensions))
            for i in range(0, self.dimensions):
                univariate_samples = self.parameters[i].getSamples(m_big)
                for j in range(0, m_big):
                    pts[j, i] = univariate_samples[j]
            #wts = float(n * 1.0)/float(m_big * 1.0) * 1.0/np.sum( (super(Polylsq, self).getPolynomial(pts))**2 , 0)
            wts =  1.0/(np.sum( super(Polylsq, self).getPolynomial(pts)**2 , 0) )**2
            wts = wts * 1.0/np.sum(wts)
        else:
            raise(ValueError, 'Polylsq:__init___:: Unknown mesh! Choose between tensor, chebyshev, random or induced please.')
        if self.gradients is False:
            self.__gradientsFalse(pts, wts, m_refined)
        elif self.gradients is True:
            self.__gradientsTrue(pts, wts, m_refined)  
    def __gradientsTrue(self, pts, wts, m_refined):
        P = super(Polylsq, self).getPolynomial(pts)
        W = np.mat( np.diag(np.sqrt(wts)) )
        A = W * P.T
        if self.optimization.lower() == 'greedy-qr':    
            __, __, pvec = qr(A.T, pivoting=True)
            z = pvec[0:m_refined]
        elif self.optimization.lower() == 'greedy-svd':   
            __, __, V = svd(A.T)
            __, __, pvec = qr(V[:, 0:m_refined].T , pivoting=True )
            z = pvec[0:m_refined]
        elif self.optimization.lower() == 'newton':
            zhat, L, ztilde, Utilde = maxdet(A, m_refined)
            z = binary2indices(zhat)
        else:
            raise(ValueError, 'Polylsq:__init___:: Unknown optimization technique! Choose between greedy or newton please.')
        index = 1
        rows = 1e15
        rank = 0
        # While loop trick to get the least number of rows in Az subject to some oversampling!
        while  rank < self.basis.cardinality:
            z_chosen = z[0:index]
            refined_pts = pts[z_chosen]
            Pz = super(Polylsq, self).getPolynomial(refined_pts)
            wts_orig_normalized =  wts[z_chosen] / np.sum(wts[z_chosen])
            Wz = np.mat(np.diag( np.sqrt(wts_orig_normalized) ) )
            Az =  Wz * Pz.T
            dPcell = super(Polylsq, self).getPolynomialGradient(refined_pts)
            dP = cell2matrix(dPcell, Wz)
            M = np.vstack([Az, dP])
            rank = np.linalg.matrix_rank(M)
            print 'Iterating...system rank: '+str(rank)+', and the # of rows in Az: '+str(index)
            index = index + 1
            del dPcell, dP 
        index = int(np.round(self.oversampling * (index - 1) ) )
        z_chosen = z[0:index]
        refined_pts = pts[z_chosen]
        Pz = super(Polylsq, self).getPolynomial(refined_pts)
        wts_orig_normalized =  wts[z_chosen] / np.sum(wts[z_chosen])
        self.Wz = np.mat(np.diag( np.sqrt(wts_orig_normalized) ) )
        self.Az =  self.Wz * Pz.T
        dPcell = super(Polylsq, self).getPolynomialGradient(refined_pts)
        self.Cz = cell2matrix(dPcell, self.Wz)  
        self.quadraturePoints = refined_pts
        self.quadratureWeights = np.sqrt(wts_orig_normalized) 
    def __gradientsFalse(self, pts, wts, m_refined):
        if self.optimization.lower() == 'random':
            m = 1.0
            n = self.basis.cardinality
            for g in range(0, self.dimensions):
                m = m * ( self.parameters[g].order + 1)
            random_samples = np.random.choice(int(m), m_refined, replace=False)
            orders_plus_one = [x+1 for x in self.orders]
            selected_indices = np.zeros((m_refined, self.dimensions))
            for i in range(0, m_refined):
                ff = np.unravel_index(random_samples[i], dims=orders_plus_one, order='C')
                for j in range(0, self.dimensions):
                    selected_indices[i, j] = ff[j]
            points_subsampled = np.zeros((m_refined, self.dimensions))
            wts = np.ones((m_refined))
            for j in range(0, self.dimensions):
                P_j, W_j = self.parameters[j]._getLocalQuadrature()
                for i in range(0, m_refined):
                    points_subsampled[i, j] = P_j[int( selected_indices[i, j] ) ]
                    wts[i] = wts[i] * W_j[ int( selected_indices[i, j] )]
            wts_orig_normalized = wts / np.sum(wts)
            #nondimensional_points_subsampled = super(Polylsq, self).scaleInputs(quadraturePoints_subsampled)
        else:
            #nondimensional_points = super(Polylsq, self).scaleInputs(pts)
            P = super(Polylsq, self).getPolynomial(pts)
            W = np.mat( np.diag(np.sqrt(wts)))
            A = W * P.T
            mmm, nnn = A.shape

            if self.optimization.lower() == 'greedy-qr':    
                __, __, pvec = qr(A.T, pivoting=True)
                z = pvec[0:m_refined]

            elif self.optimization.lower() == 'greedy-lu':    
                __, __, pvec = qr(A.T, pivoting=True)
                z = pvec[0:m_refined]
                
            elif self.optimization.lower() == 'greedy-svd':   
                __, __, V = svd(A.T)
                __, __, pvec = qr(V[:, 0:m_refined].T , pivoting=True )
                z = pvec[0:m_refined]

            elif self.optimization.lower() == 'newton':
                zhat, L, ztilde, Utilde = maxdet(A, m_refined)
                z = binary2indices(zhat)

            elif self.optimization.lower() == 'padua':
                last_index = 0
                counter = 0
                flag = 'ascending'
                organize_indices = np.zeros((mmm), dtype='int')
                while counter < mmm:
                    if flag == 'ascending':
                        for k in range(counter, counter + self.parameters[1].order + 1):
                            organize_indices[counter] = k
                            counter += 1
                            last_index = k
                        flag = 'descending'
                    else:
                        for k in range(last_index + self.parameters[1].order + 1, last_index, -1):
                            organize_indices[counter] = k
                            counter += 1
                            last_index = k
                        flag = 'ascending'
                z = organize_indices[1::2]
                
            elif self.optimization.lower() == 'none':
                z = np.arange(0, mmm, 1)

            else:
                raise(ValueError, 'Polylsq:__init___:: Unknown optimization technique! Choose between greedy or newton please.')

            points_subsampled = pts[z,:]
            #quadraturePoints_subsampled = pts[z,:]
            wts_orig_normalized =  wts[z] / np.sum(wts[z]) # if we pick a subset of the weights, they should add up to 1.!
            self.A = A
        Pz = super(Polylsq, self).getPolynomial(points_subsampled)
        Wz = np.mat(np.diag( np.sqrt(wts_orig_normalized) ) )
        self.Az =  Wz * Pz.T
        self.Wz = Wz
        self.quadraturePoints = points_subsampled
        self.quadratureWeights = wts_orig_normalized
    def quadraturePointsWeights(self):
        """
        Returns the quadrature points and weights.

        :param Polylsq self:
            An instance of the Polylsq class.
        :return:
            The quadrature points.
        :return:
            The quadrature weights.
        """
        return self.quadraturePoints, self.quadratureWeights
    def computeCoefficients(self, func, gradfunc=None, gradientmethod=None):
        """
        Computes the coefficients of the polynomial via least squares.

        :param Polylsq self:
            An instance of the Polylsq class.
        :param: callable func:
            The function that needs to be approximated. In the absence of a callable function, the input can be the function evaluated at the quadrature points.
        :param: callable gradfunc:
            The gradient of the function that needs to be approximated. In the absence of a callable gradient function, the input can be a matrix of gradient evaluations at the quadrature points.
        :param: string gradientmethod:
            The underlying strategy used to estimate the coefficients when gradient evaluations are provided. Options include:
            'stacked', 'constrained-DE', 'constrained-NS'.

        """
        # If there are no gradients, solve via standard least squares!
        if self.gradients is False:
            p, q = self.Wz.shape
            # Get function values!
            if callable(func):
                #scaled_points = super(Polylsq, self).scaleInputs(self.quadraturePoints)
                y = evalfunction(self.quadraturePoints, func)
            else:
                y = func
            self.functionEvaluations = y
            self.bz = np.dot( self.Wz ,  np.reshape(y, (p,1)) )
            alpha = np.linalg.lstsq(self.Az, self.bz) 
            self.coefficients = alpha[0]
        # If there are gradients then use a constrained least squares approach!
        elif self.gradients is True and gradfunc is not None:
            p, q = self.Wz.shape
            # Get function values!
            if callable(func):
                y = evalfunction(self.quadraturePoints, func)
            else:
                y = func
            # Get gradient values!
            if callable(func):
                grad_values = evalgradients(self.quadraturePoints, gradfunc, 'matrix')
            else:
                grad_values = gradfunc
            # Assemble gradients into a single long vector called dy!     
            p, q = grad_values.shape
            d = np.zeros((p*q,1))
            counter = 0
            for j in range(0,q):
                for i in range(0,p):
                    d[counter] = self.Wz[i,i] * grad_values[i,j]
                    counter = counter + 1
            self.dy = d
            del d, grad_values
            self.bz = np.dot( self.Wz ,  np.reshape(y, (p,1)) )
            coefficients, cond = solveCLSQ(self.Az, self.bz, self.Cz, self.dy, gradientmethod)
            self.coefficients = coefficients
        elif self.gradients is True and gradfunc is None:
            raise(ValueError, 'Polylsq:computeCoefficients:: Gradient function evaluations must be provided, either a callable function or as vectors.')
        super(Polylsq, self).__setCoefficients__(self.coefficients)
        super(Polylsq, self).__setQuadrature__(self.quadraturePoints, self.quadratureWeights)
    def getDesignMatrix(self):
        """
        Sets the design matrix.

        :param Polylsq self:
            An instance of the Polylsq class.
        """
        super(Polylsq, self).__setDesignMatrix__(self.Az)

################################
# Private functions!
################################
def rowNormalize(A):
    rows, cols = A.shape
    row_norms = np.mat(np.zeros((rows, 1)), dtype='float64')
    Normalization = np.mat(np.eye(rows), dtype='float64')
    for i in range(0, rows):
        temp = 0.0
        for j in range(0, cols):
            row_norms[i] = temp + A[i,j]**2
            temp = row_norms[i]
        row_norms[i] = (row_norms[i] * 1.0/np.float64(cols))**(-1)
        Normalization[i,i] = row_norms[i]
    A_normalized = np.dot(Normalization, A)
    return A_normalized, Normalization
