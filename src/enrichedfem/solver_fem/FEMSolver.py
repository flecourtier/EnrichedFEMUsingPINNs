print_time = False
relative_error = True
compute_H1norm = False

###########
# Imports #
###########

from enrichedfem.fenics_expressions.fenics_expressions import get_uex_expr
from enrichedfem.solver_fem.utils import get_utheta_fenics_onV
import dolfin as df

import abc
import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

latex_activate = False
try:
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    latex_activate = True
except:
    print("Latex not activated")

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 10

prm = df.parameters["krylov_solver"]
prm["absolute_tolerance"] = 1e-13
prm["relative_tolerance"] = 1e-13

current = Path(__file__).parent.parent

#######
# FEM #
#######

class FEMSolver(abc.ABC):
    """FEM solver.

    This class sets up the FEM domain, solves the FEM system,
    and computes the error between the FEM solution and the
    analytical or reference solution.  It also includes the
    enriched FEM methods (correction with addition and multiplication).
    If needed, it can compute the reference solution and save it.

    Args:
        params (list): Sample of sets of parameters.
        problem (Problem): Problem considered.
        degree (int, optional): Degree of the finite element space. Defaults to 1.
        error_degree (int, optional): Degree of the error space. Defaults to 4.
        high_degree (int, optional): Degree of the expression space for f. Defaults to 9.
        save_uref (str, optional): Directory to save reference solution. Defaults to None.
        load_uref (bool, optional): Load flag for reference solution. Defaults to True.
    """
    def __init__(self,params,problem,degree=1,error_degree=4,high_degree=9,save_uref=None,load_uref=True):
        self.N = None # number of cells
        self.params = params # list of 1 set of parameters
        self.pb_considered = problem # problem considered
        self.degree = degree # degree of the finite element space
        self.error_degree = error_degree # degree of the error space
        self.high_degree = high_degree # degree of the expression space for f
        self.save_uref = save_uref # directory to save results
        self.tab_uref = None # reference solution
        
        # To evaluate computational time
        self.times_fem = {}
        self.times_corr_add = {}
        self.times_corr_mult = {}
        
        # To compute error (overrefined mesh)
        if self.error_degree is not None:
            self.N_ex = 500 #5*self.N
            start = time.time()
            self.mesh_ex,self.V_ex,self.dx_ex = self._create_FEM_domain(self.N_ex+1,self.error_degree) 
            self.h_ex = self.mesh_ex.hmax()
            end = time.time()
            print("V_ex created with ",self.N_ex+1," vertices and degree ",self.error_degree," : h_ex =",self.h_ex)
            if print_time:
                print("Time to generate V_ex: ", end-start)
            # self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)
        
        # To create reference solution
        if not self.pb_considered.ana_sol:
            assert self.save_uref is not None and len(save_uref)==len(params)
            self.N_ref = 999
            self.error_ref = 3
            self.mesh_ref,self.V_ref,_ = self._create_FEM_domain(self.N_ref+1,self.error_ref) 
            print("V_ref created with ",self.N_ref+1," vertices and degree ",self.error_ref)
            self.load_uref = load_uref
            self.tab_uref = [self.get_uref(i) for i in range(len(self.params))]
            
    @abc.abstractmethod
    def _create_mesh(self,nb_vert):
        """Create the mesh.

        This method creates the mesh with the given number of vertices.
        For non-rectangular domains, we consider a box containing the domain.

        Args:
            nb_vert (int): Number of vertices.

        Returns:
            tuple: Mesh and computational time.
        """
        pass    

    @abc.abstractmethod
    def _define_fem_system(self,params,u,v,V_solve):
        """Define the FEM system.

        This method defines the FEM system for the given parameters,
        trial function, test function, and function space.

        Args:
            params (list): List of parameters.
            u (TrialFunction): Trial function.
            v (TestFunction): Test function.
            V_solve (FunctionSpace): Function space.

        Returns:
            tuple: Bilinear and linear forms.
        """
        pass
    
    @abc.abstractmethod
    def _define_corr_add_system(self, params, u, v, u_PINNs, V_solve):
        """Define the additive correction system.

        This method defines the additive correction system for the given parameters,
        trial function, test function, PINNs solution, and function space.

        Args:
            params (list): List of parameters.
            u (TrialFunction): Trial function.
            v (TestFunction): Test function.
            u_PINNs (Function): PINNs solution.
            V_solve (FunctionSpace): Function space.

        Returns:
            tuple: Bilinear and linear forms.
        """
        pass
    
    @abc.abstractmethod
    def _define_corr_mult_system(self, params, u, v, u_PINNs, V_solve, M, impose_bc):
        """Define the multiplicative correction system.

        This method defines the multiplicative correction system for the given parameters,
        trial function, test function, PINNs solution, function space,
        lifting constant, and boundary condition flag (weak or strong).

        Args:
            params (list): List of parameters.
            u (TrialFunction): Trial function.
            v (TestFunction): Test function.
            u_PINNs (Function): PINNs solution.
            V_solve (FunctionSpace): Function space.
            M (float): Lifting constant.
            impose_bc (bool): Boundary condition flag. If True, strong BC. If False, weak BC.

        Returns:
            tuple: Bilinear and linear forms.
        """
        pass
    
    def set_meshsize(self, nb_cell, plot_mesh=False, filename=None):
        """Set the mesh size.

        This method sets the mesh size with the given number of cells
        and creates the associated finite element space.
        It also plots the mesh if plot_mesh is True or saves the plot if filename is not None.

        Args:
            nb_cell (int): Number of cells.
            plot_mesh (bool, optional): Plot the mesh. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.
        """
        self.N = nb_cell # number of cells
        
        self.times_fem[self.N] = {}
        self.times_corr_add[self.N] = {}
        self.times_corr_mult[self.N] = {}
        
        # To compute the solution with FEM (standard/correction)
        self.mesh,self.V,self.dx = self._create_FEM_domain(self.N+1,self.degree,save_times=True)
        self.h = self.mesh.hmax()
        print("V created with ",self.N+1," vertices and degree ",self.degree," : h =",self.h)
        
        if self.high_degree is not None:
            self.V_theta = df.FunctionSpace(self.mesh, "CG", self.high_degree)
            print("V_theta created with ",self.N+1," vertices and degree ",self.high_degree)
        
        if plot_mesh or filename is not None:
            assert self.pb_considered.dim in [1,2] # to modify for 2D
            self.__plot_mesh(plot_mesh, filename)
        
    def __plot_mesh(self, plot_mesh=False, filename=None):
        """Plot the mesh.

        This method plots the mesh and saves it to a file if a filename is provided.

        Args:
            plot_mesh (bool, optional): Plot the mesh. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.
        """
        plt.figure()
        df.plot(self.mesh)
        plt.suptitle("Mesh with nb_vert="+str(self.N+1))
    
        if filename is not None:
            plt.savefig(filename)
            
        if plot_mesh:
            plt.show()
            
        plt.close()
        
        
    def _create_FEM_domain(self, nb_vert, degree, save_times=False):
        """Create the FEM domain.

        This method creates the FEM domain with the given number of vertices and degree.

        Args:
            nb_vert (int): Number of vertices.
            degree (int): Degree of the finite element space.
            save_times (bool, optional): Save computational times. Defaults to False.

        Returns:
            tuple: Mesh, function space, and integration measure.
        """   
        # Construct a cartesian mesh with nb_vert-1 cells in each direction
        mesh,tps = self._create_mesh(nb_vert)

        if save_times:
            if print_time:
                print("Time to generate mesh: ", tps)
            self.times_fem[self.N]["mesh"] = tps
            self.times_corr_add[self.N]["mesh"] = tps
            self.times_corr_mult[self.N]["mesh"] = tps
        
        V = df.FunctionSpace(mesh, "CG", degree)
        dx = df.Measure("dx", domain=mesh)

        return mesh, V, dx
    
    def run_uref(self, i):
        """Compute the reference solution.

        This method computes the reference solution for the given parameter index.

        Args:
            i (int): Parameter index.

        Returns:
            Function: Reference solution.
        """
        assert not self.pb_considered.ana_sol
        params = self.params[i]
        
        u = df.TrialFunction(self.V_ref)
        v = df.TestFunction(self.V_ref)
        
        # Declaration of the variationnal problem
        A,L = self._define_fem_system(params,u,v,self.V_ref)

        # Resolution of the linear system
        sol = df.Function(self.V_ref)
        df.solve(A,sol.vector(),L, "cg","hypre_amg")

        return sol
    
    def get_uref(self, i):
        """Get the reference solution.

        This method gets the reference solution for the given parameter index.
        It loads the solution from a file if it exists and load_uref is True,
        otherwise it computes the solution and saves it to a file.

        Args:
            i (int): Parameter index.

        Returns:
            Function: Reference solution interpolated on V_ex.
        """       
        filename = self.save_uref[i]
        
        if not self.load_uref or not os.path.exists(filename):
            print("Computing reference solution for parameter ",i)
            u_ref = self.run_uref(i)
            vct_u_ref = u_ref.vector().get_local()
            np.save(filename, vct_u_ref)  
        else:
            print("Load reference solution for parameter ",i)
            vct_u_ref = np.load(filename)
            u_ref = df.Function(self.V_ref)
            u_ref.vector()[:] = vct_u_ref
            
        u_ref_Vex = df.interpolate(u_ref,self.V_ex)
        
        return u_ref_Vex
    
    def _plot_results_fem(self, u_ex_V, sol_V, V_solve, plot_result=False, filename=None):
        """Plot the FEM results.

        This method plots the FEM solution, the analytical solution, and the error.
        It also saves the plots to a file if a filename is provided.

        Args:
            u_ex_V (Function): Analytical solution.
            sol_V (Function): FEM solution.
            V_solve (FunctionSpace): Function space.
            plot_result (bool, optional): Plot the results. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.
        """
        assert self.pb_considered.dim in [1,2]
        # Définir les tailles pour les titres et les légendes
        title_size = 24  # Taille des titres
        
        if self.pb_considered.dim == 1:
            legend_size = 20  # Taille des légendes
        
            plt.figure(figsize=(15,5))
            
            plt.subplot(1,3,2)
            if latex_activate:
                titles = [r"$u_h$",r"$u$",r"$|u-u_h|$"]
            else:
                titles = ["uh","u","|u-uh|"]
            df.plot(sol_V,label=titles[0])
            df.plot(u_ex_V,label=titles[1])
            plt.title("FEM solution",fontsize=title_size)
            plt.legend(fontsize=legend_size)
            
            plt.subplot(1,3,3)
            error_sol = df.Function(V_solve)
            error_sol.vector()[:] = abs(sol_V.vector()[:] - u_ex_V.vector()[:])
            df.plot(error_sol,label=titles[2])
            plt.title("FEM error",fontsize=title_size)
            plt.legend(fontsize=legend_size)
            
        else:
            labelsize = 14
            
            colormap = "jet"
            plt.figure(figsize=(15,5))
            
            plt.subplot(1,3,1)
            c = df.plot(u_ex_V, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} Analytical solution \\ $u$ \end{center}"
            else:
                title = "Analytical solution u"
            plt.title(title,fontsize=title_size, pad=40)
            
            plt.subplot(1,3,2)
            c = df.plot(sol_V, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center}FEM solution \\ $u_h$ \end{center}"
            else:
                title = "FEM solution uh"
            plt.title(title,fontsize=title_size, pad=40)
            
            plt.subplot(1,3,3)
            error_sol = df.Function(V_solve)
            error_sol.vector()[:] = abs(sol_V.vector()[:] - u_ex_V.vector()[:])
            c = df.plot(error_sol, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center}Error \\ $|u-u_h|$ \end{center}"
            else:
                title = "Error |u-uh|"
            plt.title(title,fontsize=title_size, pad=40)
            
        if filename is not None:
            plt.tight_layout()
            plt.gca().set_rasterization_zorder(-1)
            plt.savefig(filename,bbox_inches='tight',format="pdf")
            
        if plot_result:
            plt.show()
            
        plt.close()
        
    def fem(self, i, plot_result=False, filename=None):
        """Solve the problem using the finite element method.

        This method solves the problem for the given parameter index using the finite element method.
        It computes the solution, the error, and optionally plots the results.

        Args:
            i (int): Parameter index.
            plot_result (bool, optional): Plot the results. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.

        Returns:
            tuple: FEM solution and L2 norm of the error.
        """
        
        assert self.N is not None
        params = self.params[i]
        
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        # Declaration of the variationnal problem
        start = time.time()        
        A,L = self._define_fem_system(params,u,v,self.V)
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem[self.N]["assemble"] = end-start

        # Resolution of the linear system
        start = time.time()
        sol = df.Function(self.V)
        df.solve(A,sol.vector(),L)
        end = time.time()                
        
        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem[self.N]["solve"] = end-start

        # Compute the error
        start = time.time()
        if self.pb_considered.ana_sol:
            u_ex = get_uex_expr(params, degree=self.high_degree, domain=self.mesh_ex, pb_considered=self.pb_considered)
            uex_Vex = df.interpolate(u_ex,self.V_ex)
        else:
            uex_Vex = self.tab_uref[i]
        sol_Vex = df.interpolate(sol,self.V_ex)
        norme_L2 = (df.assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) 
        if compute_H1norm:
            norme_H1 = df.errornorm(uex_Vex, sol_Vex, norm_type='H1')
            print("norme_H1 (abs) = ",norme_H1)
        if relative_error:
            norme_L2 = norme_L2 / (df.assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
            if compute_H1norm:
                relH1 = df.norm(uex_Vex, norm_type='H1')
                norme_H1 = norme_H1 / relH1
                print("norme_H1 (rel) = ",norme_H1)            
        end = time.time()
        
        if print_time:
            print("Time to compute the error :",end-start)
        self.times_fem[self.N]["error"] = end-start
        
        if plot_result or filename is not None:
            assert self.pb_considered.dim in [1,2] # to modify for 2D
            if self.pb_considered.ana_sol:
                u_ex = get_uex_expr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
                u_ex_Vex = df.interpolate(u_ex,self.V_ex) 
            else:
                u_ex_Vex = self.tab_uref[i]
            sol_Vex = df.interpolate(sol,self.V_ex)
            self._plot_results_fem(u_ex_Vex, sol_Vex, self.V_ex, plot_result, filename)
        
        return sol,norme_L2
    
    def _plot_results_corr(self, u_ex_V, C_ex_V, C_tild_V, sol_V, V_solve, type, plot_result=False, filename=None, impose_bc=None):
        """Plot the correction results.

        This method plots the corrected solution, the analytical solution, the correction, and the errors.
        It also saves the plots to a file if a filename is provided.

        Args:
            u_ex_V (Function): Analytical solution.
            C_ex_V (Function): Analytical correction.
            C_tild_V (Function): Computed correction.
            sol_V (Function): Corrected solution.
            V_solve (FunctionSpace): Function space.
            type (str): Type of correction ("Add" or "Mult").
            plot_result (bool, optional): Plot the results. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.
            impose_bc (bool, optional): Whether boundary conditions are imposed strongly or weakly (for multiplicative correction). Defaults to None.
        """
        assert self.pb_considered.dim in [1,2]
        assert type in ["Add","Mult"]
        if type == "Mult":
            assert impose_bc is not None
            supp = " (strong)" if impose_bc else " (weak)"
        else:
            supp = ""
        
        # Définir les tailles pour les titres et les légendes
        title_size = 24  # Taille des titres
        
        if self.pb_considered.dim == 1:
            legend_size = 20  # Taille des légendes
            plt.figure(figsize=(15,5))
            
            if latex_activate:
                if type == "Add":
                    titles = [r"$p_h^+$",r"$u-u_\theta$",r"$u_h^+$",r"$u$",r"$|u-u_h^+|$"]
                else:
                    titles = [r"$p_h^\times$",r"$u/u_\theta$",r"$u_h^\times$",r"$u$",r"$|u-u_h^\times|$"]
            else:
                if type == "Add":
                    titles = ["ph+","u-utheta","uh+","u","|u-uh|"]
                else:
                    titles = ["phx","u/utheta","uhx","u","|u-uhx|"]
            
            plt.subplot(1,3,1)
            df.plot(C_tild_V,label=titles[0])
            df.plot(C_ex_V,label=titles[1])
            plt.title(f"{type} correction{supp}",fontsize=title_size)
            plt.legend(fontsize=legend_size)
            
            plt.subplot(1,3,2)
            df.plot(sol_V,label=titles[2])
            df.plot(u_ex_V,label=titles[3])
            plt.title(f"{type} solution{supp}",fontsize=title_size)
            plt.legend(fontsize=legend_size)
            
            plt.subplot(1,3,3)
            if type == "Add":
                error_sol = df.Function(V_solve)
                error_sol.vector()[:] = abs(sol_V.vector()[:] - u_ex_V.vector()[:])
            else:
                error_sol = df.Function(V_solve)
                error_sol.vector()[:] = abs(sol_V.vector()[:] - u_ex_V.vector()[:])
            df.plot(error_sol,label=titles[4])
                
            plt.title(f"{type} error{supp}",fontsize=title_size)
            plt.legend(fontsize=legend_size)
        else:
            assert type == "Add"
            labelsize = 14
            
            colormap = "jet"
            plt.figure(figsize=(15,10))
            
            # Solution
            plt.subplot(2,3,1)
            c = df.plot(u_ex_V, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} Analytical solution \\ $u$ \end{center}"
            else:
                title = "Analytical solution u"
            plt.title(title,fontsize=title_size, pad=40)
            
            plt.subplot(2,3,2)
            c = df.plot(sol_V, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} FEM solution \\ $u_h^+$ \end{center}"
            else:
                title = "FEM solution uh+"
            plt.title(title,fontsize=title_size, pad=40)
            
            plt.subplot(2,3,3)
            error_sol = df.Function(V_solve)
            error_sol.vector()[:] = abs(sol_V.vector()[:] - u_ex_V.vector()[:])
            c = df.plot(error_sol, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} Error \\ $|u-u_h^+|$ \end{center}"
            else:
                title = "Error |u-uh+|"
            plt.title(title,fontsize=title_size, pad=40)
            
            # Correction
            plt.subplot(2,3,4)
            c = df.plot(C_ex_V, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} Analytical correction \\ $u - u_\theta$ \end{center}"
            else:
                title = "Analytical correction u-utheta"
            plt.title(title,fontsize=title_size, pad=40)
            
            plt.subplot(2,3,5)
            c = df.plot(C_tild_V, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} Add correction \\ $p_h^+$ \end{center}"
            else:
                title = "Add correction ph+"
            plt.title(title,fontsize=title_size, pad=40)
            
            plt.subplot(2,3,6)
            error_C = df.Function(V_solve)
            error_C.vector()[:] = abs(C_tild_V.vector()[:] - C_ex_V.vector()[:])
            c = df.plot(error_C, cmap=colormap)
            cbar = plt.colorbar(c)
            cbar.ax.yaxis.set_tick_params(labelsize=labelsize)
            if latex_activate:
                title = r"\begin{center} Error \\ $|(u-u_\theta)-p_h^+|$ \end{center}"
            else:
                title = "Error |(u-utheta)-ph+|"
            plt.title(title,fontsize=title_size, pad=40)
            
        if plot_result:
            plt.show()
            
        if filename is not None:
            plt.tight_layout()
            plt.gca().set_rasterization_zorder(-1)
            plt.savefig(filename,bbox_inches='tight',format="pdf")
            
        plt.close()

    def pinns(self, i, u_PINNs):
        """Compute the L2 norm of the error between the PINNs solution and the analytical solution.

        This method computes the L2 norm of the error between the PINNs solution and the analytical solution
        for the given parameter index.

        Args:
            i (int): Parameter index.
            u_PINNs (Function): PINNs solution.

        Returns:
            float: L2 norm of the error.
        """
        
        assert self.N is not None
        params = self.params[i]
        
        u_theta_Vex = get_utheta_fenics_onV(self.V_ex,self.params[i],u_PINNs)
        
        if self.pb_considered.ana_sol:
            u_ex = get_uex_expr(params, degree=self.high_degree, domain=self.mesh_ex, pb_considered=self.pb_considered)
            uex_Vex = df.interpolate(u_ex,self.V_ex)
        else:
            uex_Vex = self.tab_uref[i]
        
        norme_L2 = (df.assemble((((uex_Vex - u_theta_Vex)) ** 2) * self.dx) ** (0.5)) 
        if relative_error:
            norme_L2 = norme_L2 / (df.assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return norme_L2
        
    def corr_add(self, i, u_PINNs, plot_result=False, filename=None):
        """Solve the problem using the additive correction method.

        This method solves the problem for the given parameter index using the additive correction method.
        It computes the solution, the correction, the error, and optionally plots the results.

        Args:
            i (int): Parameter index.
            u_PINNs (Function): PINNs solution.
            plot_result (bool, optional): Plot the results. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.

        Returns:
            tuple: Corrected solution, correction, and L2 norm of the error.
        """
        
        assert self.N is not None
        params = self.params[i]
        
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        # Declaration of the variationnal problem
        start = time.time()
        A,L = self._define_corr_add_system(params,u,v,u_PINNs,self.V)
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_corr_add[self.N]["assemble"] = end-start

        # Resolution of the linear system
        start = time.time()
        C_tild = df.Function(self.V)
        df.solve(A,C_tild.vector(),L)
        
        sol = df.Function(self.V)
        u_theta_V = get_utheta_fenics_onV(self.V,self.params[i],u_PINNs)      
        sol.vector()[:] = C_tild.vector()[:] + u_theta_V.vector()[:]
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add[self.N]["solve"] = end-start

        # Compute the error
        start = time.time()
        u_theta_Vex = get_utheta_fenics_onV(self.V_ex,self.params[i],u_PINNs)
        if self.pb_considered.ana_sol:
            u_ex = get_uex_expr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
            uex_Vex = df.interpolate(u_ex,self.V_ex) 
        else:
            uex_Vex = self.tab_uref[i]
        C_Vex = df.interpolate(C_tild,self.V_ex)
        sol_Vex = df.Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+u_theta_Vex.vector()[:]

        
        norme_L2 = (df.assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) 
        if relative_error:
            norme_L2 = norme_L2 / (df.assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        end = time.time()
        
        if compute_H1norm:
            norme_H1 = df.errornorm(uex_Vex, sol_Vex, norm_type='H1')
            print("norme_H1 = ",norme_H1)
            if relative_error:
                relH1 = df.norm(uex_Vex, norm_type='H1')
                norme_H1 = norme_H1 / relH1
                print("norme_H1 (rel) = ",norme_H1)
        
        if print_time:
            print("Time to compute the error :",end-start)
        self.times_corr_add[self.N]["error"] = end-start
        
        if plot_result or filename is not None:
            if self.pb_considered.ana_sol:
                u_ex = get_uex_expr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
                u_ex_Vex = df.interpolate(u_ex,self.V_ex) 
            else:
                u_ex_Vex = self.tab_uref[i]
            C_ex_Vex = df.Function(self.V_ex)
            C_ex_Vex.vector()[:] = u_ex_Vex.vector()[:] - u_theta_Vex.vector()[:]
            self._plot_results_corr(u_ex_Vex,C_ex_Vex,C_Vex,sol_Vex,self.V_ex,type="Add",filename=filename)
        
        return sol,C_tild,norme_L2

    def corr_mult(self, i, u_PINNs, M=0.0, impose_bc=True, plot_result=False, filename=None):
        """Solve the problem using the multiplicative correction method.

        This method solves the problem for the given parameter index using the multiplicative correction method.
        It computes the solution, the correction, the error, and optionally plots the results.

        Args:
            i (int): Parameter index.
            u_PINNs (Function): PINNs solution.
            M (float, optional): Lifting constant. Defaults to 0.0.
            impose_bc (bool, optional): Whether boundary conditions are imposed strongly or weakly. Defaults to True.
            plot_result (bool, optional): Plot the results. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to None.

        Returns:
            tuple: Corrected solution, correction, and L2 norm of the error.
        """
        
        assert self.N is not None
        params = self.params[i]
        self.times_corr_mult[self.N][str(M)] = {}
        
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        # Declaration of the variationnal problem
        start = time.time()
        A,L = self._define_corr_mult_system(params,u,v,u_PINNs,self.V,M,impose_bc=impose_bc)
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_corr_mult[self.N][str(M)]["assemble"] = end-start

        # Resolution of the linear system
        start = time.time()
        C_tild = df.Function(self.V)
        df.solve(A,C_tild.vector(),L)
        
        sol = df.Function(self.V)
        u_theta_V = get_utheta_fenics_onV(self.V,self.params[i],u_PINNs)      
        sol.vector()[:] = C_tild.vector()[:] * (u_theta_V.vector()[:] + M) - M
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_mult[self.N][str(M)]["solve"] = end-start

        # Compute the error
        start = time.time()
        u_theta_Vex = get_utheta_fenics_onV(self.V_ex,self.params[i],u_PINNs)
        if self.pb_considered.ana_sol:
            u_ex = get_uex_expr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
            uex_Vex = df.interpolate(u_ex,self.V_ex) 
        else:
            uex_Vex = self.tab_uref[i]
        C_Vex = df.interpolate(C_tild,self.V_ex)
        sol_Vex = df.Function(self.V_ex)
        sol_Vex.vector()[:] = C_Vex.vector()[:] * (u_theta_Vex.vector()[:] + M) - M
        print("on fait du mult")
        
        norme_L2 = (df.assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) 
        if relative_error:
            norme_L2 = norme_L2 / (df.assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        end = time.time()
        
        if compute_H1norm:
            norme_H1 = df.errornorm(uex_Vex, sol_Vex, norm_type='H1')
            print("norme_H1 = ",norme_H1)
            if relative_error:
                relH1 = df.norm(uex_Vex, norm_type='H1')
                norme_H1 = norme_H1 / relH1
                print("norme_H1 (rel) = ",norme_H1)        
        
        if print_time:
            print("Time to compute the error :",end-start)
        self.times_corr_mult[self.N][str(M)]["error"] = end-start
        
        if plot_result or filename is not None:
            assert self.pb_considered.dim == 1 # to modify for 2D
            u_ex_Vex = df.interpolate(u_ex,self.V_ex)
            C_ex_Vex = df.Function(self.V_ex)
            C_ex_Vex.vector()[:] = np.divide(u_ex_Vex.vector()[:], u_theta_Vex.vector()[:], out=np.full_like(u_ex_Vex.vector()[:], np.nan), where=u_theta_Vex.vector()[:] != 0)
            self._plot_results_corr(u_ex_Vex,C_ex_Vex,C_Vex,sol_Vex,self.V_ex,type="Mult",filename=filename,impose_bc=impose_bc)
        
        return sol,C_tild,norme_L2
