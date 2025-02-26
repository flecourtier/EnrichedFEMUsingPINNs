###########
# Imports #
###########

import dolfin as dol
from dolfin.function.expression import (
    BaseExpression,
    _select_element,
    _InterfaceExpression,
)

import sympy as sp
import dolfin as df

#######################
# get_expr_from_sympy #
#######################

def get_expr_from_sympy(params, degree, domain, fct):
    """Convert a symbolic function to a FEniCS expression.

    This function takes a symbolic function defined using SymPy and converts
    it into a FEniCS expression that can be used within FEniCS computations.
    It handles parameter substitution and conversion to C++ code for JIT compilation.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        fct (callable): The symbolic function to convert, which should take
            SymPy, spatial variables, and parameters as arguments.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the input symbolic function.
    """
    # Crée les symboles pour les variables
    dim = domain.geometric_dimension()
    xy = sp.symbols(' '.join(['xx', 'yy', 'zz'][:dim]))
    # xy = symbols_list if dim > 1 else symbols_list[0]
    
    # Crée les symboles pour les paramètres
    nb_parameters = len(params)
    params_dict = {f'p{i+1}': params[i] for i in range(nb_parameters)}
    mu = sp.symbols(' '.join(params_dict.keys()))
    mu = (mu,) if nb_parameters == 1 else mu
    
    # Crée le dictionnaire des paramètres pour df.Expression
    expression_params = {str(symbol): value for symbol, value in zip(mu, params_dict.values())}
    
    # Remplace 'xx' par 'x[0]' et 'yy' par 'x[1]' pour l'expression dans df.Expression
    fct_sympy = fct(sp, xy, mu)
    fct_fe = df.Expression(sp.ccode(fct_sympy).replace('log','std::log').replace('xx', 'x[0]').replace('yy', 'x[1]'),degree=degree, domain=domain, **expression_params)
    
    return fct_fe

def get_uex_expr(params, degree, domain, pb_considered):
    """Convert the symbolic exact solution to a FEniCS expression.

    This function converts the symbolic exact solution (`u_ex`) of the
    considered problem to a FEniCS expression. It uses the
    `get_expr_from_sympy` function to perform the conversion.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `u_ex` attribute representing the symbolic exact solution.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the exact solution.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.u_ex)

def get_f_expr(params, degree, domain, pb_considered):
    """Convert the symbolic source term to a FEniCS expression.

    This function converts the symbolic source term (`f`) of the considered
    problem to a FEniCS expression. It uses the `get_expr_from_sympy`
    function to perform the conversion.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `f` attribute representing the symbolic source term.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the source term.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.f)

def get_g_expr(params, degree, domain, pb_considered):
    """Convert the symbolic Dirichlet boundary condition to a FEniCS expression.

    This function converts the symbolic Dirichlet boundary condition (`g`) of
    the considered problem to a FEniCS expression. It uses the
    `get_expr_from_sympy` function to perform the conversion.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `g` attribute representing the symbolic Dirichlet boundary
            condition.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the Dirichlet boundary condition.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.g)

def get_gn_expr(params, degree, domain, pb_considered):
    """Convert the symbolic Neumann boundary condition to a FEniCS expression.

    This function converts the symbolic Neumann boundary condition (`g`) of
    the considered problem to a FEniCS expression. It uses the
    `get_expr_from_sympy` function to perform the conversion.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `g` attribute representing the symbolic Neumann boundary
            condition.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the Neumann boundary condition.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.gn)

def get_gr_expr(params, degree, domain, pb_considered):
    """Convert the symbolic Robin boundary condition to a FEniCS expression.

    This function converts the symbolic Robin boundary condition (`g`) of
    the considered problem to a FEniCS expression. It uses the
    `get_expr_from_sympy` function to perform the conversion.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `g` attribute representing the symbolic Robin boundary
            condition.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the Robin boundary condition.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.gr)

def get_h_ext_expr(params, degree, domain, pb_considered):
    """Convert the symbolic exterior boundary condition to a FEniCS expression.

    This function converts the symbolic exterior boundary condition
    (`h_ext`) of the considered problem to a FEniCS expression. It uses the
    `get_expr_from_sympy` function to perform the conversion. Used in the donut problem.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `h_ext` attribute representing the symbolic exterior
            boundary condition.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the exterior boundary condition.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.h_ext)

def get_h_int_expr(params, degree, domain, pb_considered):
    """Convert the symbolic interior boundary condition to a FEniCS expression.

    This function converts the symbolic interior boundary condition
    (`h_int`) of the considered problem to a FEniCS expression. It uses the
    `get_expr_from_sympy` function to perform the conversion. Used in the donut problem.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have a
            `h_int` attribute representing the symbolic interior
            boundary condition.

    Returns:
        dolfin.Expression: The FEniCS expression corresponding to the interior boundary condition.
    """
    return get_expr_from_sympy(params, degree, domain, pb_considered.h_int)

######################
# MyUserExpression22 #
######################

#  for 2x2 matrices (used for TestCase 3)
class MyUserExpression22(BaseExpression):
    """Custom FEniCS expression for 2x2 matrices.

    This class extends FEniCS's `BaseExpression` to represent 2x2 matrix-valued
    expressions. It facilitates the definition and evaluation of such
    expressions within FEniCS, enabling JIT compilation for efficient
    computation.

    Args:
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
    """

    def __init__(self, degree, domain):
        cell = domain.ufl_cell()
        element = _select_element(
            family=None, cell=cell, degree=degree, value_shape=(2,2)
        )

        self._cpp_object = _InterfaceExpression(self, (2,2))

        BaseExpression.__init__(
            self,
            cell=cell,
            element=element,
            domain=domain,
            name=None,
            label=None,
        )
        
class AnisotropyExpr(MyUserExpression22):
    """FEniCS expression for an anisotropic diffusion matrix.

    This class represents a FEniCS expression for a 2x2 anisotropic diffusion
    matrix. It inherits from `MyUserExpression22` to handle JIT compilation
    and evaluation of the matrix at given spatial points.

    Args:
        params (list or tuple): The parameters of the problem.
        degree (int): The degree of the finite element.
        domain (dolfin.Mesh): The mesh of the problem.
        pb_considered: The problem being considered, which should have an
            `anisotropy_matrix` method for defining the matrix symbolically.
    """
    def __init__(self, params, degree, domain, pb_considered):
        super().__init__(degree, domain)
        self.mu = params
        self.pb_considered = pb_considered

    def eval(self, value, x):
        """Evaluate the anisotropy matrix at a given point.

        This method evaluates the 2x2 anisotropy matrix at a given spatial
        point `x` and stores the result in the `value` array.

        Args:
            value (numpy.ndarray): The array to store the evaluated matrix.
            x (tuple): The spatial coordinates of the point.
        """
        val = self.pb_considered.anisotropy_matrix(dol, x, self.mu)
        value[0] = val[0]
        value[1] = val[1]
        value[2] = val[2]
        value[3] = val[3]