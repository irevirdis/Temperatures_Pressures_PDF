"""The polynomial parent class."""
import numpy as np
from .stats import Statistics

class Poly(object):
    """
    The class defines a Poly object. It is the parent class to Polyreg, Polyint and Polycs; the only difference between its children are the way in which the coefficients are computed. This class is defined by a list of Parameter objects and a Basis.

    :param Parameter parameters:
        A list of parameters.
    :param Basis basis:
        A basis selected for the multivariate polynomial.

    """
    def __init__(self, parameters, basis):
        try:
            len(parameters)
        except TypeError:
            parameters = [parameters]
        self.parameters = parameters
        self.basis = basis
        self.dimensions = len(parameters)
        self.orders = []
        if not self.basis.orders :
            for i in range(0, self.dimensions):
                self.orders.append(self.parameters[i].order)
            self.basis.setOrders(self.orders)

    def __setCoefficients__(self, coefficients):
        """
        Sets the coefficients for polynomial. This function will be called by the children of Poly

        :param Poly self:
            An instance of the Poly class.
        :param array coefficients:
            An array of the coefficients computed using either integration, least squares or compressive sensing routines.

        """
        self.coefficients = coefficients
    def __setBasis__(self, basisNew):
        """
        Sets the basis
        """
        self.basis = basisNew 
    def __setQuadrature__(self, quadraturePoints, quadratureWeights):
        """
        Sets the quadrature points and weights

        :param Poly self:
            An instance of the Poly class.
        :param matrix quadraturePoints:
            A numpy matrix filled with the quadrature points.
        :param matrix quadratureWeights:
            A numpy matrix filled with the quadrature weights.
        """
        self.quadraturePoints = quadraturePoints
        self.quadratureWeights = quadratureWeights
    def __setDesignMatrix__(self, designMatrix):
        """
        Sets the design matrix assocaited with the quadrature (depending on the technique) points and the polynomial basis.

        :param Poly self:
            An instance of the Poly class.
        :param matrix designMatrix:
            A numpy matrix filled with the multivariate polynomial evaluated at the quadrature points.

        """
        self.designMatrix = designMatrix
    def clone(self):
        """
        Clones a Poly object.

        :param Poly self:
            An instance of the Poly class.
        :return:
            A clone of the Poly object.
        """
        return type(self)(self.parameters, self.basis)
    def getPolynomial(self, stackOfPoints, customBases=None):
        """
        Evaluates the multivariate polynomial at a set of points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the multivarite (in d-dimensions) polynomial must be evaluated.
        :return:
            A N-by-1 matrix of polynomial evaluations at the stackOfPoints.
        """
        if customBases is None:
            basis = self.basis.elements
        else:
            basis = customBases
        basis_entries, dimensions = basis.shape
        no_of_points, _ = stackOfPoints.shape
        polynomial = np.zeros((basis_entries, no_of_points))
        p = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            poly , _ =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis)))
            return poly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , _ = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i])) )

        # One loop for polynomials
        for i in range(0, basis_entries):
            temp = np.ones((1, no_of_points))
            for k in range(0, dimensions):
                polynomial[i,:] = p[k][int(basis[i,k])] * temp
                temp = polynomial[i,:]

        return polynomial
    def getPolynomialGradient(self, stackOfPoints):
        """
        Evaluates the gradient of the multivariate polynomial at a set of points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points along which the multivarite (in d-dimensions) polynomial must be evaluated.
        :return:
            A list with d elements, each with a N-by-1 matrix of polynomial evaluations at the stackOfPoints.
        """
        # "Unpack" parameters from "self"
        basis = self.basis.elements
        basis_entries, dimensions = basis.shape
        no_of_points, _ = stackOfPoints.shape
        p = {}
        dp = {}

        # Save time by returning if univariate!
        if dimensions == 1:
            _ , dpoly =  self.parameters[0]._getOrthoPoly(stackOfPoints, int(np.max(basis) ) )
            return dpoly
        else:
            for i in range(0, dimensions):
                if len(stackOfPoints.shape) == 1:
                    stackOfPoints = np.array([stackOfPoints])
                p[i] , dp[i] = self.parameters[i]._getOrthoPoly(stackOfPoints[:,i], int(np.max(basis[:,i]) + 1 ) )

        # One loop for polynomials
        R = []
        for v in range(0, dimensions):
            gradDirection = v
            polynomialgradient = np.zeros((basis_entries, no_of_points))
            for i in range(0, basis_entries):
                temp = np.ones((1, no_of_points))
                for k in range(0, dimensions):
                    if k == gradDirection:
                        polynomialgradient[i,:] = dp[k][int(basis[i,k])] * temp
                    else:
                        polynomialgradient[i,:] = p[k][int(basis[i,k])] * temp
                    temp = polynomialgradient[i,:]
            R.append(polynomialgradient)

        return R
    def getTensorQuadratureRule(self, orders=None):
        """
        Generates a tensor grid quadrature rule based on the parameters in Poly.

        :param Poly self:
            An instance of the Poly class.
        :param list orders:
            A list of the highest polynomial orders along each dimension.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        # Initialize points and weights
        pp = [1.0]
        ww = [1.0]

        if orders is None:
            orders = self.basis.orders

        # number of parameters
        # For loop across each dimension
        for u in range(0, self.dimensions):

            # Call to get local quadrature method (for dimension 'u')
            local_points, local_weights = self.parameters[u]._getLocalQuadrature(orders[u])
            ww = np.kron(ww, local_weights)

            # Tensor product of the points
            dummy_vec = np.ones((len(local_points), 1))
            dummy_vec2 = np.ones((len(pp), 1))
            left_side = np.array(np.kron(pp, dummy_vec))
            right_side = np.array( np.kron(dummy_vec2, local_points) )
            pp = np.concatenate((left_side, right_side), axis = 1)

        # Ignore the first column of pp
        points = pp[:,1::]
        weights = ww

        # Return tensor grid quad-points and weights
        return points, weights
    def getStatistics(self, light=None, max_sobol_order=None):
        """
        Creates an instance of the Statistics class.

        :param Poly self:
            An instance of the Poly class.
        :param string quadratureRule:
            Two options exist for this string. The user can use 'qmc' for a distribution specific Monte Carlo (QMC) or they can use 'tensor grid' for standard tensor product grid. Typically, if the number of dimensions is less than 8, the tensor grid is the default option selected.
        :return:
            A Statistics object.
        """
        if light is None:
            evals = self.getPolynomial(self.quadraturePoints)
            return Statistics(self.coefficients, self.basis, self.parameters, self.quadraturePoints, self.quadratureWeights, evals, max_sobol_order)
        else:
            return Statistics(self.coefficients, self.basis, self.parameters, max_sobol_order=max_sobol_order)            
    def getQuadratureRule(self, options=None, number_of_points = None):
        """
        Generates quadrature points and weights.

        :param Poly self:
            An instance of the Poly class.
        :param string options:
            Two options exist for this string. The user can use 'qmc' for a distribution specific Monte Carlo (QMC) or they can use 'tensor grid' for standard tensor product grid. Typically, if the number of dimensions is less than 8, the tensor grid is the default option selected.
        :param int number_of_points:
            If QMC is chosen, specifies the number of quadrature points in each direction. Otherwise, this is ignored.
        :return:
            A numpy array of quadrature points.
        :return:
            A numpy array of quadrature weights.
        """
        if options is None:
            if self.dimensions > 5 or np.max(self.orders) > 4:
                options = 'qmc'
            else:
                options = 'tensor grid'
        if options.lower() == 'qmc':
            if number_of_points is None:
                default_number_of_points = 20000
            else:
                default_number_of_points = number_of_points
            p = np.zeros((default_number_of_points, self.dimensions))
            w = 1.0/float(default_number_of_points) * np.ones((default_number_of_points))
            for i in range(0, self.dimensions):
                p[:,i] = np.array(self.parameters[i].getSamples(m=default_number_of_points)).reshape((default_number_of_points,))
            return p, w

        if options.lower() == 'tensor grid' or options.lower() == 'quadrature':
            p,w = self.getTensorQuadratureRule([i for i in self.basis.orders])
            return p,w
    def evaluatePolyFit(self, stackOfPoints):
        """
        Evaluates the the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A 1-by-N matrix of the polynomial approximation.
        """
        return self.getPolynomial(stackOfPoints).T *  np.mat(self.coefficients)

    def evaluatePolyGradFit(self, stackOfPoints):
        """
        Evaluates the gradient of the polynomial approximation of a function (or model data) at prescribed points.

        :param Poly self:
            An instance of the Poly class.
        :param matrix stackOfPoints:
            A N-by-d matrix of points (can be unscaled) at which the polynomial gradient must be evaluated at.
        :return:
            A d-by-N matrix of the gradients of the polynomial approximation.

        **Notes:**

        This function should not be confused with getPolynomialGradient(). The latter is only concerned with approximating what the multivariate polynomials
        gradient values are at prescribed points.
        """
        H = self.getPolynomialGradient(stackOfPoints)
        grads = np.zeros((self.dimensions, len(stackOfPoints) ) )
        for i in range(0, self.dimensions):
            grads[i,:] = np.mat(self.coefficients).T * H[i]
        return grads
    def getPolyFitFunction(self):
        """
        Returns a callable polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.

        """
        return lambda (x): self.getPolynomial(x).T *  np.mat(self.coefficients)
    def getPolyGradFitFunction(self):
        """
        Returns a callable for the gradients of the polynomial approximation of a function (or model data).

        :param Poly self:
            An instance of the Poly class.
        :return:
            A callable function.

        """
        return lambda (x) : self.evaluatePolyGradFit(x)
    def getFunctionSamples(self, number_of_samples):
        """
        Returns a set of function samples; useful for computing probabilities.

        :param Poly self:
            An instance of the Poly class.
        :param callable function:
            A callable function (or evaluations of the function at the prerequisite quadrature points).
        :param array coefficients:
            A numpy array of the coefficients
        :param matrix indexset:
            A K-by-d matrix of the index set.
        :return:
            A 50000-by-1 array of function evaluations.

        """
        dimensions = self.dimensions
        if number_of_samples is None:
            number_of_samples = 50000 # default value!
        plotting_pts = np.zeros((number_of_samples, dimensions))
        for i in range(0, dimensions):
                univariate_samples = self.parameters[i].getSamples(number_of_samples)
                for j in range(0, number_of_samples):
                    plotting_pts[j, i] = univariate_samples[j]
        samples = self.evaluatePolyFit(plotting_pts)
        return plotting_pts, samples
