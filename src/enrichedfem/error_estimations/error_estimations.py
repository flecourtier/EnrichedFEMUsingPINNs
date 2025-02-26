import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from enrichedfem.problem.utils import create_tree,select_param,compute_slope
from enrichedfem.error_estimations.utils import get_solver_type

class ErrorEstimations:
    """Error estimates class for the FEM method and enriched approaches.

    This class allows to run the error estimates for the FEM method and enriched approaches (additive and multiplicative corrections). The L2 error is computed for different degrees and number of vertices.
    The results are saved in csv files and can be plotted.

    Args:
        param_num (int): Number of the parameter to consider
        pb_considered (Problem): Problem to consider        
        error_degree (int, optional): Degree of the error space. Defaults to 4.
        high_degree (int, optional): Degree of the expression space for f. Defaults to 9.
        save_fig (bool, optional): Save flag for the figures. Defaults to False.
        plot_result (bool, optional): Plot flag for the results. Defaults to False.
        plot_mesh (bool, optional): Plot flag for the mesh. Defaults to False.
        tab_nb_vert (list, optional): List of the number of vertices. Defaults to [2**i for i in range(4,9)].
        tab_degree (list, optional): List of the degrees. Defaults to [1,2,3].
    """
    def __init__(self, param_num, pb_considered, **kwargs):
        # define the problem
        self.param_num = param_num
        self.pb_considered = pb_considered
        self.__infos_from_problem()
        repo_dir = kwargs.get('repo_dir', "./")
        version_str = f"version{self.version}"
        
        self.results_dir = repo_dir + f"/results/fenics/test_{self.dim}D/testcase{self.testcase}/{version_str}/cvg/param{param_num}/"
        create_tree(self.results_dir)
        
        print(f"## Results directory: {self.results_dir}")
        
        # define the degrees and the number of vertices
        self.error_degree = kwargs.get('error_degree', 4)
        self.high_degree = kwargs.get('high_degree', 10)
        self.save_fig = kwargs.get('save_fig', False)
        self.plot_result = kwargs.get('plot_result', False)
        self.plot_mesh = kwargs.get('plot_mesh', False)
        self.tab_nb_vert = kwargs.get('tab_nb_vert', [2**i for i in range(4,9)])
        self.tab_degree = kwargs.get('tab_degree', [1,2,3])
        
        # define if the reference solution is an analytical solution
        self.save_uref = None
        if not self.pb_considered.ana_sol:
            savedir = self.results_dir + "u_ref/"
            create_tree(savedir)
            filename = savedir + f"u_ref_{param_num}.npy"
            self.save_uref = [filename]
            
        
    def __infos_from_problem(self):
        """Extract problem information.

        This method extracts information about the problem, such as dimension,
        testcase, version, parameters, and solver type, from the
        `pb_considered` attribute and stores them as attributes of the class.
        """
        self.dim = self.pb_considered.dim
        self.testcase = self.pb_considered.testcase
        self.version = self.pb_considered.version
        self.params = [select_param(self.pb_considered,self.param_num)]     
        self.solver_type = get_solver_type(self.dim,self.testcase,self.version)
    
    def read_csv(self,csv_file):
        """Read a CSV file.

        :param csv_file: CSV file to read
        :return: DataFrame, list of h values, list of error values
        """
        df_meth = pd.read_csv(csv_file)
        tab_h_meth = list(df_meth['h'].values)
        tab_err_meth = list(df_meth['err'].values)
        
        tab_nb_vert_meth = list(df_meth['nb_vert'].values)
        assert all(tab_nb_vert_meth) == all(self.tab_nb_vert)
        
        return df_meth, tab_h_meth, tab_err_meth
    
    def run_error_estimations_deg(self, method, degree, **kwargs):
        """Run error estimates for a given method and degree.

        This method performs error estimates using the specified method
        (FEM, Corr, or Mult) for a given degree. It reads results from a CSV
        file if available, otherwise it runs the error estimation and saves
        the results to a CSV file.

        Args:
            method (str): The error estimation method ("FEM", "Corr", or "Mult").
            degree (int): The degree of the finite element solution.
            new_run (bool): Whether to force a new run even if a CSV file exists.
            u_theta: Network prediction. Required for "Corr" and "Mult" methods.
            M (float): Lifting constant. Required for "Mult" method.
            impose_bc (bool): Required for "Mult" method.

        Returns:
            tuple: A tuple containing the DataFrame, list of h values, and list
                of error values.
        """
        new_run = kwargs.get('new_run', False)
        assert method in ["FEM","Corr","Mult"], f"method={method} is not implemented"
        
        assert "u_theta" in kwargs if method in ["Corr","Mult"] else True, f"u_theta is required for {method}"
        u_theta = kwargs.get('u_theta')
        
        if method == "Mult":
            assert 'M' in kwargs and 'impose_bc' in kwargs, f"M and impose_bc are required for {method}"
            M = kwargs.get('M')
            impose_bc = kwargs.get('impose_bc')
        else:
            assert not 'M' in kwargs and not 'impose_bc' in kwargs, f"M and impose_bc are not required for {method}"

        # Define the csv_file        
        csv_file = self.results_dir+f"{method}_case{self.testcase}_v{self.version}_param{self.param_num}_degree{degree}"
        if method == "Mult":
            csv_file += f"_M{M}"
            csv_file += '_weak' if not impose_bc else ''
        csv_file += ".csv"
        
        # Read the csv file if it exists (or if not new_run)
        if not new_run and os.path.exists(csv_file):
            print(f"## Read csv file : {csv_file}")
            return self.read_csv(csv_file)

        # Run the error estimation
        print(f"## Run error estimation with {method} for degree={degree}")
        tab_h_method = []
        tab_err_method = []
        solver = self.solver_type(params=self.params, problem=self.pb_considered, degree=degree, error_degree=self.error_degree, high_degree=self.high_degree, save_uref=self.save_uref)
        for nb_vert in self.tab_nb_vert:
            mesh_filename = None
            if self.plot_mesh:
                results_dir_mesh = self.results_dir + f"Mesh_plot/"
                create_tree(results_dir_mesh)
                mesh_filename = results_dir_mesh+f"Mesh_plot_case{self.testcase}_v{self.version}_N{nb_vert}.png"
            solver.set_meshsize(nb_cell=nb_vert-1,plot_mesh=self.plot_mesh,filename=mesh_filename)
            tab_h_method.append(solver.h)
            fig_filename = None
            
            if self.save_fig:
                results_dir_fig = self.results_dir + f"{method}_plot"
                if method == "Mult" and not impose_bc:
                    results_dir_fig += '_weak'
                results_dir_fig += "/"
                create_tree(results_dir_fig)
                fig_filename = results_dir_fig+f"{method}_plot_case{self.testcase}_v{self.version}_param{self.param_num}_degree{degree}_N{nb_vert}"
                if method == "Mult":
                    fig_filename += f"_M{M}"
                    if not impose_bc:
                        fig_filename += '_weak'
                fig_filename += '.pdf'
            
            if method == "FEM": 
                _,norme_L2 = solver.fem(0,plot_result=self.plot_result,filename=fig_filename)
            elif method == "Corr":
                _,_,norme_L2 = solver.corr_add(0,u_theta,plot_result=self.plot_result,filename=fig_filename)
            else:
                _,_,norme_L2 = solver.corr_mult(0,u_theta,M=M,impose_bc=impose_bc,plot_result=self.plot_result,filename=fig_filename)
                
            print(f"nb_vert={nb_vert}, norme_L2={norme_L2}")
            tab_err_method.append(norme_L2)
            
        df_method = pd.DataFrame({'nb_vert': self.tab_nb_vert, 'h': tab_h_method, 'err': tab_err_method})
        df_method.to_csv(csv_file, index=False)
                
        return df_method, tab_h_method, tab_err_method
    
    def run_error_estimations_alldeg(self, method, **kwargs):
        """Run error estimates for a given method and all degrees.

        This method runs the error estimates for the given method and all
        degrees specified in `self.tab_degree`. It saves the results to a CSV
        file and optionally plots the convergence results.

        Args:
            method (str): The error estimation method ("FEM", "Corr", or "Mult").
            **kwargs: Additional keyword arguments. These may include:
                plot_cvg (bool): Whether to plot the convergence results.
                u_theta: Network prediction. Required for "Corr" and "Mult" methods.
                M (float): Lifting constant. Required for "Mult" method.
                impose_bc (bool): Required for "Mult" method.
                new_run (bool): Whether to force a new run even if a CSV file exists.

        Returns: None
        """
        plot_cvg = kwargs.get('plot_cvg', False)

        if method == "Mult":
            assert 'M' in kwargs and 'impose_bc' in kwargs, f"M and impose_bc are required for {method}"
            M = kwargs.get('M')
            impose_bc = kwargs.get('impose_bc')
        else:
            assert not 'M' in kwargs and not 'impose_bc' in kwargs, f"M and impose_bc are not required for {method}"

        # Define the csv_file
        csv_file_alldeg = self.results_dir+f'{method}_case{self.testcase}_v{self.version}_param{self.param_num}'
        if method == "Mult":
            csv_file_alldeg += f"_M{M}"
            csv_file_alldeg += '_weak' if not impose_bc else ''
        csv_file_alldeg += ".csv"
        
        # Run the error estimation for all degrees
        dict_alldeg = {}
        for degree in self.tab_degree:
            _, _, tab_err_method = self.run_error_estimations_deg(method,degree,**kwargs)
            
            # to save
            if degree == 1:
                dict_alldeg['N'] = self.tab_nb_vert
            dict_alldeg[f'P{degree}'] = tab_err_method
            
        df_deg = pd.DataFrame(dict_alldeg)
        df_deg.to_csv(csv_file_alldeg, index=False)
        
        if plot_cvg:
            plt.figure(figsize=(5, 5))
            
            for degree in self.tab_degree:
                tab_err_method = dict_alldeg[f'P{degree}']
                
                # to plot
                if plot_cvg:
                    plt.loglog(self.tab_nb_vert, tab_err_method, "+-", label='P'+str(degree))
                
                    for i in range(1,len(self.tab_nb_vert)):
                        slope, vert_mid = compute_slope(i,self.tab_nb_vert,tab_err_method)
                        plt.text(vert_mid[0]+1e-2 , vert_mid[1], str(slope), fontsize=12, ha='left', va='top')

            plt.xticks(self.tab_nb_vert, np.array(self.tab_nb_vert).round(3).astype(str))
            plt.xlabel('nb_vert')
            plt.ylabel('L2 norm')
            if method != "Mult":
                title = f'{method} case{self.testcase} v{self.version} param{self.param_num} : {self.params[0]}'
                fig_filename = self.results_dir+f'{method}_case{self.testcase}_v{self.version}_param{self.param_num}.png'
            else:
                type = "(weak) " if not impose_bc else ""
                title = f'{method} {type}case{self.testcase} v{self.version} param{self.param_num} : {self.params[0]} (M={M})'
                type = "_weak" if not impose_bc else ""
                fig_filename = self.results_dir+f'{method}_case{self.testcase}_v{self.version}_param{self.param_num}_M{M}{type}.png'
            plt.title(title)
            plt.legend()
            plt.savefig(fig_filename)
            plt.show()
        
    
    def run_fem_deg(self, degree, new_run=False):
        """Run FEM error estimates for a given degree.

        This method runs the error estimates using the FEM method for a given
        degree. It calls the `run_error_estimations_deg` method with the
        specified parameters.

        Args:
            degree (int): The degree of the finite element solution.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.

        Returns:
            tuple: A tuple containing the DataFrame, list of h values, and list
                of error values.
        """
        return self.run_error_estimations_deg("FEM",degree,new_run=new_run)
    
    def run_corr_deg(self, degree, u_theta, new_run=False):
        """Run additive correction error estimates for a given degree.

        This method runs the error estimates using the additive correction
        method ("Corr") for a given degree. It calls the
        `run_error_estimations_deg` method with the specified parameters.

        Args:
            degree (int): The degree of the finite element solution.
            u_theta: The predicted solution.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.

        Returns:
            tuple: A tuple containing the DataFrame, list of h values, and list
                of error values.
        """
        return self.run_error_estimations_deg("Corr",degree,u_theta=u_theta,new_run=new_run)
    
    def run_mult_deg_M(self, degree, u_theta, M=0.0, impose_bc=True, new_run=False):
        """Run multiplicative correction error estimates for a given degree and M value.

        This method runs the error estimates using the multiplicative correction
        method ("Mult") for a given degree and M value. It calls the
        `run_error_estimations_deg` method with the specified parameters.

        Args:
            degree (int): The degree of the finite element solution.
            u_theta: The predicted solution.
            M (float, optional): Lifting constant. Defaults to 0.0.
            impose_bc (bool, optional): Whether to impose boundary conditions. Defaults to True.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.

        Returns:
            tuple: A tuple containing the DataFrame, list of h values, and list
                of error values.
        """
        return self.run_error_estimations_deg("Mult",degree,u_theta=u_theta,M=M,impose_bc=impose_bc,new_run=new_run)
    
    
    def run_fem_alldeg(self, new_run=False, plot_cvg=False):
        """Run FEM error estimates for all degrees.

        This method runs the error estimates using the FEM method for all
        degrees specified in `self.tab_degree`. It calls the
        `run_error_estimations_alldeg` method with the specified parameters.

        Args:
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.
            plot_cvg (bool, optional): Whether to plot the convergence results. Defaults to False.

        Returns: None
        """
        self.run_error_estimations_alldeg("FEM",new_run=new_run,plot_cvg=plot_cvg)
    
    def run_corr_alldeg(self, u_theta, new_run=False, plot_cvg=False):
        """Run additive correction error estimates for all degrees.

        This method runs the error estimates using the additive correction
        method ("Corr") for all degrees specified in `self.tab_degree`.
        It calls the `run_error_estimations_alldeg` method with the
        specified parameters.

        Args:
            u_theta: The predicted solution.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.
            plot_cvg (bool, optional): Whether to plot the convergence results. Defaults to False.

        Returns: None
        """
        self.run_error_estimations_alldeg("Corr",u_theta=u_theta,new_run=new_run,plot_cvg=plot_cvg)
    
    def run_mult_alldeg_M(self, u_theta, M=0.0, impose_bc=True, new_run=False, plot_cvg=False):
        """Run multiplicative correction error estimates for all degrees and a given M value.

        This method runs the error estimates using the multiplicative correction
        method ("Mult") for all degrees specified in `self.tab_degree` and a
        given M value. It calls the `run_error_estimations_alldeg` method
        with the specified parameters.

        Args:
            u_theta: The predicted solution.
            M (float, optional): Lifting constant. Defaults to 0.0.
            impose_bc (bool, optional): Whether to impose boundary conditions. Defaults to True.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.
            plot_cvg (bool, optional): Whether to plot the convergence results. Defaults to False.

        Returns: None
        """
        self.run_error_estimations_alldeg("Mult",u_theta=u_theta,M=M,impose_bc=impose_bc,new_run=new_run,plot_cvg=plot_cvg)
        
    def run_mult_deg_allM(self, degree, u_theta, tab_M, impose_bc=True, new_run=False):
        """Run multiplicative correction error estimates for a given degree and multiple M values.

        This method runs the error estimates using the multiplicative correction
        method ("Mult") for a given degree and a range of M values.

        Args:
            degree (int): The degree of the finite element solution.
            u_theta: The predicted solution.
            tab_M (list): A list of M values to use in the multiplicative error estimation.
            impose_bc (bool, optional): Whether to impose boundary conditions. Defaults to True.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.

        Returns: None
        """
        for M in tab_M:
            self.run_mult_deg_M(degree,u_theta,M=M,impose_bc=impose_bc,new_run=new_run)
            
    def run_mult_alldeg_allM(self, u_theta, tab_M, impose_bc=True, new_run=False, plot_cvg=False):
        """Run multiplicative correction error estimates for all degrees and multiple M values.

        This method runs the error estimates using the multiplicative correction
        method ("Mult") for all degrees specified in `self.tab_degree` and a
        range of M values.

        Args:
            u_theta: The predicted solution.
            tab_M (list): A list of M values to use in the multiplicative error estimation.
            impose_bc (bool, optional): Whether to impose boundary conditions. Defaults to True.
            new_run (bool, optional): Whether to force a new run even if a CSV file exists. Defaults to False.
            plot_cvg (bool, optional): Whether to plot the convergence results. Defaults to False.

        Returns: None
        """
        for M in tab_M:
            self.run_mult_alldeg_M(u_theta,M=M,impose_bc=impose_bc,new_run=new_run,plot_cvg=plot_cvg)