import pandas as pd
import numpy as np
import dataframe_image as dfi

from enrichedfem.modfenics.gains.gains import GainsEnhancedFEM

class CompareGainsMethods:
    """Compare gains of enhanced FEM methods.

    This class compares the gains achieved by enhanced FEM methods (additive
    and multiplicative corrections) over standard FEM and PINNs. It reads
    error data from CSV files, computes gains, creates dataframes for errors
    and gains, and saves statistics about the gains.

    Args:
        gains_enhanced_fem (GainsEnhancedFEM): An instance of the
            `GainsEnhancedFEM` class, containing the error data.
    """
    def __init__(self,gains_enhanced_fem:GainsEnhancedFEM):
        self.gef = gains_enhanced_fem
        self.params_str = self.create_params_str()
        self.__row_names = [str(i) + " : " + self.params_str[i] for i in range(self.gef.n_params)]
        
    def create_params_str(self):
        """Create string representations of parameter sets.

        This method generates a list of strings, where each string represents
        a set of parameters used in the problem. The parameters are rounded
        to two decimal places and separated by commas within each string.

        Returns:
            list: A list of strings, each representing a parameter set.
        """
        dim_params = len(self.gef.params[0])
        params_str = []
        for i in range(self.gef.n_params):
            param_str = ""
            for j in range(dim_params):
                param_str += f"{self.gef.params[i][j].round(2)}"
                if j < dim_params-1:
                    param_str += ","
            params_str.append(param_str)
        return params_str

    
    def __read_errors(self, degree, tab_M=None):
        """Read error data for different methods.

        This method reads error data from CSV files for FEM, PINNs, additive
        correction ("Corr"), and multiplicative correction ("Mult") methods
        for a given degree and a list of M values. It raises
        FileNotFoundError if any of the required files are not found.

        Args:
            degree (int): The degree of the finite element solution.
            tab_M (list, optional): A list of M values to consider for the "Mult" methods. Defaults to None.

        Returns:
            tuple: A tuple containing a dictionary of error arrays for each
                method and a list of mesh sizes (h).

        Raises:
            FileNotFoundError: If any of the required error files are not found.
        """            
        tab_nb_vert = self.gef.tab_nb_vert
        
        # Read date for all methods
        tab_errors = {}
        try:
            csv_file = self.gef.results_dir+f'FEM_errors_case{self.gef.testcase}_v{self.gef.version}_degree{degree}.csv'
            print(csv_file)
            _,tab_h,tab_errors["FEM"] = self.gef.read_csv(csv_file)
        except:
            tab_errors["FEM"] = None
            raise FileNotFoundError(f'FEM P{degree} not found')        

        assert tab_errors["FEM"] is not None, "FEM errors not found"

        try:
            csv_file = self.gef.results_dir+f'PINNs_errors_case{self.gef.testcase}_v{self.gef.version}_degree{degree}.csv'
            _,_,tab_errors["PINNs"] = self.gef.read_csv(csv_file)
        except:
            tab_errors["PINNs"] = None
            raise FileNotFoundError(f'PINNs P{degree} not found')
        
        assert tab_errors["PINNs"] is not None, "PINNs errors not found"
    
        try:
            csv_file = self.gef.results_dir+f'Corr_errors_case{self.gef.testcase}_v{self.gef.version}_degree{degree}.csv'
            _,_,tab_errors["Corr"] = self.gef.read_csv(csv_file)
        except:
            tab_errors["Corr"] = None
            raise FileNotFoundError(f'Corr P{degree} not found')
        
        dict_Mult = {}
        dict_Mult_weak = {}
        if tab_M is not None:
            for M in tab_M:
                try:
                    csv_file = self.gef.results_dir+f'Mult_errors_case{self.gef.testcase}_v{self.gef.version}_degree{degree}_M{M}.csv'
                    _,_,tab_err_Mult = self.gef.read_csv(csv_file)
                    dict_Mult[M] = tab_err_Mult
                except:
                    print(f'Mult strong P{degree} M{M} not found')
                
                try:
                    csv_file = self.gef.results_dir+f'Mult_errors_case{self.gef.testcase}_v{self.gef.version}_degree{degree}_M{M}_weak.csv'
                    _,_,tab_err_Mult = self.gef.read_csv(csv_file)
                    dict_Mult_weak[M] = tab_err_Mult
                except:
                    print(f'Mult weak P{degree} M{M} not found')
        
            assert len(dict_Mult) in [0,len(tab_M)], "Number of Mult methods is not correct"
            assert len(dict_Mult_weak) in [0,len(tab_M)], "Number of Mult weak methods is not correct"

        tab_errors["Mult"] = dict_Mult
        tab_errors["Mult_weak"] = dict_Mult_weak
        
        return tab_errors,tab_h

    def create_dferrors_deg_allM(self, degree, tab_M=None):
        """Create a DataFrame of errors for a given degree and all M values.

        This method reads error data for different methods (FEM, PINNs,
        Corr, Mult, Mult_weak) from CSV files, and constructs a pandas
        DataFrame where each row represents a parameter sample and each
        column represents a method, mesh size, and mesh size (h). The DataFrame
        is then saved to a CSV file.

        Args:
            degree (int): The degree of the finite element solution.
            tab_M (list, optional): A list of M values to consider for the "Mult" methods. Defaults to None.

        Returns:
            pandas.DataFrame: The DataFrame of errors.
        """
        tab_nb_vert = self.gef.tab_nb_vert
        size = len(tab_nb_vert)
        
        tab_errors,tab_h = self.__read_errors(degree,tab_M=tab_M)
        
        # Create dataframe for errors
        col_names = []
        for method in tab_errors.keys():
            tab_errors_method = tab_errors[method]
            if tab_errors_method is not None and len(tab_errors_method) > 0:
                if "Mult" in method:
                    for M in tab_M:
                        for i in range(size):
                            col_names += [(method+str(M),str(tab_nb_vert[i]),str(tab_h[i]))]
                else:
                    for i in range(size):
                        col_names += [(method,str(tab_nb_vert[i]),str(tab_h[i]))]
        
        mi = pd.MultiIndex.from_tuples(col_names, names=["method","n_vert","h"])
        df_errors = pd.DataFrame(columns=mi,index=self.__row_names)
        
        for i in range(self.gef.n_params):
            col = 0
            for j in range(size):
                df_errors.loc[self.__row_names[i],col_names[col+j]] = tab_errors["FEM"][i,j]
            col += size
            for j in range(size):
                df_errors.loc[self.__row_names[i],col_names[col+j]] = tab_errors["PINNs"][i,j]
            col += size
            if tab_errors["Corr"] is not None:
                for j in range(size):
                    df_errors.loc[self.__row_names[i],col_names[col+j]] = tab_errors["Corr"][i,j]
                col += size
            if tab_M is not None:
                if len(tab_errors["Mult"]) > 0:
                    for M in tab_M:
                        for j in range(size):
                            df_errors.loc[self.__row_names[i],col_names[col+j]] = tab_errors["Mult"][M][i,j]
                        col += size
                if len(tab_errors["Mult_weak"]) > 0:
                    for M in tab_M:
                        for j in range(size):
                            df_errors.loc[self.__row_names[i],col_names[col+j]] = tab_errors["Mult_weak"][M][i,j]            
                        col += size
                        
        save_file = self.gef.results_dir+f'df_errors_case{self.gef.testcase}_v{self.gef.version}_degree{degree}.csv'
        df_errors.to_csv(save_file)
    
        return df_errors
    
    def __compute_gains(self, df_errors):
        """Compute gains of enhanced methods over FEM and PINNs.

        This method computes the gains of additive ("Corr") and multiplicative
        ("Mult") correction methods over standard FEM and PINNs, based on
        the errors provided in the input DataFrame.

        Args:
            df_errors (pandas.DataFrame): A DataFrame containing the errors
                for each method, with "method" as a level in the columns
                MultiIndex.

        Returns:
            tuple: A tuple containing two dictionaries: `tab_gains_overFEM`
                with gains over FEM, and `tab_gains_overPINNs` with gains
                over PINNs.
        """
        tab_gains_overFEM = {}
        tab_gains_overPINNs = {}
        for method in df_errors.columns.get_level_values("method").unique():
            if method != "FEM":
                tab_gains_overFEM[method] = df_errors["FEM"] / df_errors[method]
                if method != "PINNs":
                    tab_gains_overPINNs[method] = df_errors["PINNs"] / df_errors[method]
        
        return tab_gains_overFEM,tab_gains_overPINNs
    
    def create_dataframes_deg_allM(self, degree, tab_M=None):
        """Create DataFrames of errors and gains for a given degree and all M values.

        This method creates two pandas DataFrames: one for errors and one
        for gains, for a given degree and all specified M values. The error
        DataFrame contains the L2 errors for each method (FEM, PINNs, Corr,
        Mult, Mult_weak), while the gains DataFrame contains the gains of
        enhanced methods over FEM and PINNs. Both DataFrames are saved to
        CSV files.

        Args:
            degree (int): The degree of the finite element solution.
            tab_M (list, optional): A list of M values to consider for the "Mult" methods. Defaults to None.

        Returns:
            pandas.DataFrame: The DataFrame of gains.
        """
        # Create dataframe for errors
        df_errors = self.create_dferrors_deg_allM(degree,tab_M=tab_M)
        tab_methods = df_errors.columns.get_level_values("method").unique()
        
        tab_nb_vert = self.gef.tab_nb_vert
        size = len(tab_nb_vert)
        tab_h = df_errors.columns.get_level_values("h").unique().to_numpy()
        
        # Compute gains
        tab_gains_overFEM,tab_gains_overPINNs = self.__compute_gains(df_errors)
    
        # Create dataframe for gains
        col_names = []
        for method in tab_methods:
            if method != "FEM":
                if method != "PINNs":
                    col_names += [(f"PINNs/{method}",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(size)]
                col_names += [(f"FEM/{method}",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(size)]
    
        mi = pd.MultiIndex.from_tuples(col_names, names=["facteurs","n_vert","h"])
        df_gains = pd.DataFrame(columns=mi,index=self.__row_names)

        # Fill dataframe for gains
        col = 0
        for method in tab_methods:
            if method != "FEM":
                if method != "PINNs":
                    for j in range(size):
                        col_name = col_names[col+j]
                        df_gains.loc[:,col_name] = tab_gains_overPINNs[method].to_numpy()[:,j]
                    col += size
                for j in range(size):
                    col_name = col_names[col+j]
                    df_gains.loc[:,col_name] = tab_gains_overFEM[method].to_numpy()[:,j]
                col += size
        
        save_file = self.gef.results_dir+f'df_gains_case{self.gef.testcase}_v{self.gef.version}_degree{degree}.csv'            
        df_gains.to_csv(save_file)
        
        return df_gains
        
    def create_dataframes_alldeg_allM(self, tab_M=None):
        """Create DataFrames of errors and gains for all degrees and all M values.

        This method creates and saves DataFrames of errors and gains for all
        degrees specified in `self.gef.tab_degree` and all specified M values.
        It calls the `create_dataframes_deg_allM` method for each degree.

        Args:
            tab_M (list, optional): A list of M values to consider for the "Mult" methods. Defaults to None.

        Returns: None
        """
        for degree in self.gef.tab_degree:
            self.create_dataframes_deg_allM(degree,tab_M=tab_M)
            
                
    def save_stats_deg_allM(self, degree, tab_M=None):
        """Save statistics of gains for a given degree and all M values.

        This method computes and saves statistics (min, max, mean, std) of the
        gains of enhanced methods (Corr, Mult, Mult_weak) over FEM and PINNs
        for a given degree and all specified M values. The statistics are
        saved as a CSV file and a PNG image.

        Args:
            degree (int): The degree of the finite element solution.
            tab_M (list, optional): A list of M values to consider for the "Mult" methods. Defaults to None.

        Returns:
            pandas.DataFrame: The DataFrame containing the rounded statistics.
        """
        df_errors = self.create_dferrors_deg_allM(degree,tab_M=tab_M)
        tab_gains_overFEM,tab_gains_overPINNs = self.__compute_gains(df_errors)
        tab_nb_vert = self.gef.tab_nb_vert   
        size = len(tab_nb_vert)   
        tab_enhmethods = list(tab_gains_overFEM.keys())
        tab_enhmethods.remove("PINNs")
        
        def get_stats(tab_gains):
            df_min = tab_gains.min(axis=0).to_numpy()
            df_max = tab_gains.max(axis=0).to_numpy()
            df_mean = tab_gains.mean(axis=0).to_numpy()
            df_std = tab_gains.std(axis=0).to_numpy()
            
            return np.array([df_min,df_max,df_mean,df_std]).T


        stats_overPINNs = []
        stats_overFEM = []

        for method in tab_enhmethods:            
            gains_overPINNs_meth = tab_gains_overPINNs[method]
            stats_overPINNs_meth = get_stats(gains_overPINNs_meth)
            stats_overPINNs.append(stats_overPINNs_meth)
            
            gains_overFEM_meth = tab_gains_overFEM[method]
            stats_overFEM_meth = get_stats(gains_overFEM_meth)
            stats_overFEM.append(stats_overFEM_meth)
            
        stats_overPINNs = np.array(stats_overPINNs).reshape(-1,4)
        stats_overFEM = np.array(stats_overFEM).reshape(-1,4)
        global_stats = np.concatenate([stats_overPINNs,stats_overFEM],axis=1)
                    
        
        tab_methods = ["PINNs", "FEM"]
        type_stats = ["min","max","mean","std"]
        col_names = []
        for method in tab_methods:
            for type_stat in type_stats:
                col_names.append((method,type_stat))
                
        row_names = []
        for method in tab_enhmethods:
            for j in range(size):
                row_names.append((method,str(tab_nb_vert[j])))

        mi = pd.MultiIndex.from_tuples(col_names, names=["method","type"])
        ri = pd.MultiIndex.from_tuples(row_names, names=["method","n_vert"])
        df_stats = pd.DataFrame(global_stats,columns=mi,index=ri)
        
        result_file = self.gef.results_dir+f'Tab_stats_case{self.gef.testcase}_v{self.gef.version}_degree{degree}'

        df_stats.to_csv(result_file+'.csv')
        
        # df_stats_round = df_stats.round(2)
        # df_stats_round = df_stats_round.astype(int)

        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        df_stats_round = df_stats.applymap(lambda x: f"{x:.2f}")
        print(df_stats_round)
        dfi.export(df_stats_round,result_file+".png",dpi=1000,table_conversion=table_conversion)
        
        return df_stats_round
    
    def save_stats_alldeg_allM(self, tab_M=None):
        """Save statistics of gains for all degrees and all M values.

        This method computes and saves statistics (min, max, mean, std) of
        the gains of enhanced methods over FEM and PINNs for all degrees
        specified in `self.gef.tab_degree` and all specified M values. It
        calls the `save_stats_deg_allM` method for each degree.

        Args:
            tab_M (list, optional): A list of M values to consider for the "Mult" methods. Defaults to None.

        Returns: None
        """
        for degree in self.gef.tab_degree:
            self.save_stats_deg_allM(degree,tab_M=tab_M)