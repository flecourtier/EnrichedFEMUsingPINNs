from enrichedfem.solver_fem.PoissonDirFEMSolver import PoissonDirSquareFEMSolver,PoissonDirDonutFEMSolver,PoissonDirLineFEMSolver
from enrichedfem.solver_fem.EllipticDirFEMSolver import EllipticDirSquareFEMSolver,Elliptic1DDirLineFEMSolver
from enrichedfem.solver_fem.PoissonMixedFEMSolver import PoissonMixedDonutFEMSolver
# from enrichedfem.solver_fem.PoissonModNeuFEMSolver import PoissonModNeuDonutFEMSolver

def get_solver_type(dim,testcase,version):
    if dim==1:
        if testcase == 1:
            return PoissonDirLineFEMSolver
        elif testcase == 2:
            return Elliptic1DDirLineFEMSolver
        else:
            pass
    elif dim==2:
        if testcase in [1,2]:
            return PoissonDirSquareFEMSolver
        elif testcase == 3:
            return EllipticDirSquareFEMSolver
        elif testcase == 4:
            return PoissonDirDonutFEMSolver
        elif testcase == 5:
            return PoissonMixedDonutFEMSolver
        else:
            pass
    else:
        pass