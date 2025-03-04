from enrichedfem.geometry.geometry_2D import Square1, UnitSquare, UnitCircle, Circle2, Donut1, Donut2
from math import *
import torch
import abc
import sympy

class TestCase2D(abc.ABC):
    """Abstract base class for 2D test cases.

    This class defines the common interface for all 2D test cases, including
    properties for geometry, number of parameters, parameter domain, and
    availability of an analytical solution.

    Args:
        testcase (int): The test case number.
        version (int): The version number of the test case.
    """
    def __init__(self,testcase,version):
        self.testcase = testcase
        self.version = version
        self.dim = 2

        @property
        @abc.abstractmethod
        def geometry(self):
            pass
        @property
        @abc.abstractmethod
        def nb_parameters(self):
            pass
        @property
        @abc.abstractmethod
        def parameter_domain(self):
            pass
        @property
        @abc.abstractmethod
        def ana_sol(self):
            pass
        
class TestCase1(TestCase2D):
    """Test case 1 for 2D problems.

    This test case defines a 2D Poisson problem with low frequency in a square domain with Dirichlet BC, 
    two parameters (mu1, mu2),
    and provides an analytical solution.

    Args:
        version (int): The version number of the test case. Defaults to 1.
    """
    def __init__(self,version=1):
        super().__init__(1,version)
        assert version in [1,2,3]
        self.geometry = Square1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]
        self.ana_sol = True
        self.predexactBC = True
        if version == 3:
            self.predexactBC = False

    def u_ex(self, pre, xy, mu):
        """Exact solution u(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The value of the exact solution at the given point.
        """
        x,y=xy
        mu1,mu2 = mu
        ex = pre.exp(-((x-mu1)**2.0 +(y-mu2)**2.0)/2)
        return ex * pre.sin(2*x) * pre.sin(2*y)

    def gradu_ex(self, pre, xy, mu):
        """Gradient of the exact solution grad(u)(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The gradient of the exact solution at the given point.
        """
        x,y=xy
        mu1,mu2 = mu
        du_dx = (-1.0*(-mu1 + x)**1.0*pre.sin(2*x) + 2.0*pre.cos(2*x))*pre.exp(-(-mu1 + x)**2.0/2 - (-mu2 + y)**2.0/2)*pre.sin(2*y)
        du_dy = (-1.0*(-mu2 + y)**1.0*pre.sin(2*y) + 2.0*pre.cos(2*y))*pre.exp(-(-mu1 + x)**2.0/2 - (-mu2 + y)**2.0/2)*pre.sin(2*x)
        return du_dx, du_dy

    def grad2u_ex(self, pre, xy, mu):
        """Hessian of the exact solution grad^2(u)(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The Hessian of the exact solution at the given point.
        """
        x,y=xy
        mu1,mu2 = mu

        d2u_dx2 = 1.0*(-4.0*(-mu1 + x)**1.0*pre.cos(2*x) + 1.0*(-mu1 + x)**2.0*pre.sin(2*x) - 5.0*pre.sin(2*x))*pre.exp(-(-mu1 + x)**2.0/2 - (-mu2 + y)**2.0/2)*pre.sin(2*y)
        d2u_dxy = (1.0*(-mu1 + x)**1.0*(-mu2 + y)**1.0*pre.sin(2*x)*pre.sin(2*y) - 2.0*(-mu1 + x)**1.0*pre.sin(2*x)*pre.cos(2*y) - 2.0*(-mu2 + y)**1.0*pre.sin(2*y)*pre.cos(2*x) + 4*pre.cos(2*x)*pre.cos(2*y))*pre.exp(-(-mu1 + x)**2.0/2 - (-mu2 + y)**2.0/2)
        d2u_dy2 = 1.0*(-4.0*(-mu2 + y)**1.0*pre.cos(2*y) + 1.0*(-mu2 + y)**2.0*pre.sin(2*y) - 5.0*pre.sin(2*y))*pre.exp(-(-mu1 + x)**2.0/2 - (-mu2 + y)**2.0/2)*pre.sin(2*x)
        return d2u_dx2, d2u_dxy, d2u_dy2

    def f(self, pre, xy, mu):
        """Source term f(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The value of the source term at the given point.
        """
        x,y=xy
        mu1,mu2 = mu
        return -pre.exp(-((x - mu1)**2 + (y - mu2)**2)/2) * (((x**2 - 2*mu1*x + mu1**2 - 5)*pre.sin(2*x) + (4*mu1 - 4*x)*pre.cos(2*x)) * pre.sin(2*y) + pre.sin(2*x) * ((y**2 - 2*mu2*y + mu2**2 - 5)*pre.sin(2*y) + (4*mu2 - 4*y)*pre.cos(2*y)))

    def gradf(self, pre, xy, mu):
        """Gradient of the source term grad(f)(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The gradient of the source term at the given point.
        """
        x,y=xy
        mu1,mu2 = mu
        exp = pre.exp((-(y - mu2) ** 2 - (x - mu1) ** 2) / 2)
        sin1 = pre.sin(2 * x)
        sin2 = pre.sin(2 * y)
        cos1 = pre.cos(2 * x)
        cos2 = pre.cos(2 * y)

        df_dx = -(mu1 - x) * exp * (
            ((mu2 - y) ** 2 + (mu1 - x) ** 2 - 10) * sin1 * sin2
            + 4 * (mu1 - x) * cos1 * sin2
            + 4 * (mu2 - y) * sin1 * cos2
        ) + exp * (
            (4 - 2 * ((mu2 - y) ** 2 + (mu1 - x) ** 2 - 10)) * cos1 * sin2
            + 10 * (mu1 - x) * sin1 * sin2
            - 8 * (mu2 - y) * cos1 * cos2
        )

        df_dy = -(mu2 - y) * exp * (
            ((mu2 - y) ** 2 + (mu1 - x) ** 2 - 10) * sin1 * sin2
            + 4 * (mu1 - x) * cos1 * sin2
            + 4 * (mu2 - y) * sin1 * cos2
        ) + exp * (
            (4 - 2 * ((mu2 - y) ** 2 + (mu1 - x) ** 2 - 10)) * sin1 * cos2
            + 10 * (mu2 - y) * sin1 * sin2
            - 8 * (mu1 - x) * cos1 * cos2
        )

        return df_dx, df_dy

    def g(self, pre, xy, mu):
        """Dirichlet boundary condition g(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The value of the Dirichlet boundary condition.
        """
        return 0.0

class TestCase2(TestCase2D):
    """Test case 2 for 2D problems.

    This test case defines a 2D Poisson problem with high frequency in a square domain with Dirichlet BC, 
    two parameters (mu1, mu2),
    and provides an analytical solution.

    Args:
        version (int): The version number of the test case. Defaults to 1.
    """
    def __init__(self,version=1):
        super().__init__(2,version)
        self.geometry = Square1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]
        self.ana_sol = True
        self.predexactBC = True

    def u_ex(self, pre, xy, mu):
        """Exact solution u(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The value of the exact solution at the given point.
        """
        x,y=xy
        mu1,mu2 = mu
        ex = pre.exp(-((x-mu1)**2 +(y-mu2)**2)/2.0)
        return ex * pre.sin(8*x) * pre.sin(8*y)

    def f(self, pre, xy, mu):
        """Source term f(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The value of the source term at the given point.
        """
        x,y=xy
        mu1,mu2 = mu

        return (16.0*(x-mu1)*pre.sin(8*y)*pre.cos(8*x) - 1.0*(x-mu1)**2.0*pre.sin(8*x)*pre.sin(8*y) + 16.0*(y-mu2)*pre.sin(8*x)*pre.cos(8*y) - 1.0*(y-mu2)**2.0*pre.sin(8*x)*pre.sin(8*y) + 130.0*pre.sin(8*x)*pre.sin(8*y))*pre.exp(-(x-mu1)**2.0/2 - (y-mu2)**2.0/2)

    def g(self, pre, xy, mu):
        """Dirichlet boundary condition g(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu1, mu2).

        Returns:
            The value of the Dirichlet boundary condition.
        """
        return 0.0
    
class TestCase3(TestCase2D):
    """Test case 3 for 2D problems.

    This test case defines a 2D anisotropic elliptic problem on a square domain with Dirichlet BC,
    four parameters (c1, c2, sigma, eps), and does not provide an analytical solution.

    Args:
        version (int): The version number of the test case. Defaults to 1.
    """
    def __init__(self,version=1):
        assert version in [1,2]
        super().__init__(3,version)
        self.geometry = UnitSquare() 
        self.nb_parameters = 4
        self.parameter_domain = [[0.4, 0.6],[0.4, 0.6],[0.1, 0.8],[0.01, 1.0]] #c1,c2,sigma,eps
        self.ana_sol = False
        self.predexactBC = True

    def u_ex(self, pre, xy, mu):
        """Exact solution u(x,y).

        Not available for this test case.

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (c1, c2, sigma, eps).

        Returns:
            None
        """
        pass

    def anisotropy_matrix(self, pre, xy, mu):
        """Anisotropy matrix A(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (c1, c2, sigma, eps).

        Returns:
            The components of the anisotropy matrix (a11, a12, a21, a22).
        """
        x,y = xy
        _,_,_, eps = mu

        a11 = eps * x**2 + y**2
        a12 = (eps - 1) * x * y
        a21 = (eps - 1) * x * y
        a22 = x**2 + eps * y**2

        return a11, a12, a21, a22

    def f(self, pre, xy, mu):
        """Source term f(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (c1, c2, sigma, eps).

        Returns:
            The value of the source term at the given point.
        """
        x,y=xy
        c1,c2,sigma,eps = mu
        return pre.exp(-((x - c1) ** 2 + (y - c2) ** 2) / (0.025 * sigma**2))

    def g(self, pre, xy, mu):
        """Dirichlet boundary condition g(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (c1, c2, sigma, eps).

        Returns:
            The value of the Dirichlet boundary condition.
        """
        return 0.0
        
class TestCase4(TestCase2D):
    """Test case 4 for 2D problems.

    This test case defines a 2D Poisson problem on an annulus with mixed boundary conditions
    (Robin on the inner boundary, Dirichlet on the outer boundary),
    one parameter (mu), and provides an analytical solution.

    Args:
        version (int): The version number of the test case. Defaults to 1.
    """
    def __init__(self,version=1):
        assert version in [1,2,3]
        super().__init__(5,version)
        self.geometry = Donut2()
        self.nb_parameters = 1
        if self.version != 3:
            self.parameter_domain = [[2.4, 2.600001]]
        else:
            self.parameter_domain = [[2.0, 3.000001]]
        self.ana_sol = True
        self.predexactBC = True

    def u_ex(self, pre, xy, mu):
        """Exact solution u(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy, torch).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The value of the exact solution at the given point.
        """
        x,y = xy
        if pre is torch:
            ln = pre.log
        else:
            ln = pre.ln

        mu = mu[0]
        return 1.0 - ln(mu * pre.sqrt(x**2 + y**2))/log(4.0)

    def gradu_ex(self, pre, xy, mu):
        """Gradient of the exact solution grad(u)(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The gradient of the exact solution at the given point.
        """
        x,y=xy

        mu = mu[0]
        du_dx = -x/((x**2 + y**2)*log(4))
        du_dy = -y/((x**2 + y**2)*log(4))
        return du_dx, du_dy

    def grad2u_ex(self, pre, xy, mu):
        """Hessian of the exact solution grad^2(u)(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The Hessian of the exact solution at the given point.
        """
        x,y=xy

        mu = mu[0]

        d2u_dx2 = -(y**2 - x**2)/((x**2 + y**2)**2*log(4))
        d2u_dxy = 2*x*y/((x**2 + y**2)**2*log(4))
        d2u_dy2 = -(x**2 - y**2)/((x**2 + y**2)**2*log(4))
        return d2u_dx2, d2u_dxy, d2u_dy2

    def f(self, pre, xy, mu):
        """Source term f(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The value of the source term at the given point.
        """
        x,y = xy
        return 0.0

    def gradf(self, pre, xy, mu):
        """Gradient of the source term grad(f)(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The gradient of the source term at the given point.
        """
        x,y = xy
        return 0.0, 0.0

    def h_int(self, pre, xy, mu):
        """Robin boundary condition on the inner boundary h_int(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The value of the Robin boundary condition on the inner boundary.
        """
        mu = mu[0]
        return (4.0-log(mu))/log(4.0) + 2.0

    def h_ext(self, pre, xy, mu):
        """Dirichlet boundary condition on the outer boundary h_ext(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The value of the Dirichlet boundary condition on the outer boundary.
        """
        mu = mu[0]
        return 1.0 - log(mu)/log(4.0)

    def gr(self, pre, xy, mu): # robin
        """Robin boundary condition gr(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The value of the Robin boundary condition.
        """
        return self.h_int(pre, xy, mu)

    def g(self, pre, xy, mu): # dirichlet
        """Dirichlet boundary condition g(x,y).

        Args:
            pre: The precision module (e.g., dolfin, numpy).
            xy: The spatial coordinate(s) (x,y).
            mu: The parameter vector (mu).

        Returns:
            The value of the Dirichlet boundary condition.
        """
        return self.h_ext(pre, xy, mu)