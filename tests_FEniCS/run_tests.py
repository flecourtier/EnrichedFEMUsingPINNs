import json
from utils import read_config
from select_tests import select_tests
import importlib
import os

new_run = False
plot_cvg = False

current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(current_dir)
if current_dir[-1] == "/":
    current_dir = current_dir[:-1]

def run_error_estimates(eeconfig,problem,u_theta):
    print("\n############################## Error Estimates ##############################\n")
    
    from enrichedfem.error_estimations.error_estimations import ErrorEstimations
    from enrichedfem.error_estimations.compare import CompareMethods
    
    if "param_num" in eeconfig:
        tab_param_num = eeconfig["param_num"]   
        tab_degree = eeconfig["degree"]  
        tab_methods = eeconfig["methods"]
        global_methods = list(tab_methods.keys())
        
        for param_num in tab_param_num:
            print(f"\n############################## Param num: {param_num} ##############################\n")
            error_estimations = ErrorEstimations(param_num, pb_considered=problem, repo_dir = current_dir, tab_degree=tab_degree)
            compare_methods = CompareMethods(error_estimations)
            
            if "FEM" in global_methods:
                print(f"\n###### FEM")
                error_estimations.run_fem_alldeg(new_run=new_run,plot_cvg=plot_cvg)
            
            if "Add" in global_methods:
                print(f"\n###### Add")
                error_estimations.run_corr_alldeg(u_theta,new_run=new_run,plot_cvg=plot_cvg)
                
                compare_methods.plot_Corr_vs_FEM_alldeg()
            
            for method in global_methods:
                if "Mult" in method:
                    print(f"\n###### {method}")
                    
                    tab_M = tab_methods[method]

                    impose_bc=True
                    if method[-1] == "W":
                        impose_bc=False
                    
                    error_estimations.run_mult_alldeg_allM(u_theta, tab_M=tab_M, impose_bc=impose_bc, new_run=new_run, plot_cvg=plot_cvg)
    
                    compare_methods.plot_Mult_vs_Add_vs_FEM_alldeg_allM(tab_M)
            
            compare_methods.save_tab_alldeg_allM(tab_M)
    
    else:
        assert False, "Given parameters are not implemented yet"
        
def run_gains(gainsconfig,problem,u_theta):
    print("\n############################## Gains ##############################\n")
    
    from enrichedfem.gains.gains import GainsEnhancedFEM
    from enrichedfem.gains.compare import CompareGainsMethods
    
    n_params = gainsconfig["n_params"]   
    tab_degree = gainsconfig["degree"]  
    tab_enmethods = gainsconfig["methods"]
    global_enmethods = list(tab_enmethods.keys())
    
    gains_enhanced_fem = GainsEnhancedFEM(n_params, problem, repo_dir = current_dir, tab_degree=tab_degree)
    compare_gains_methods = CompareGainsMethods(gains_enhanced_fem)
    
    print(f"\n###### FEM")
    gains_enhanced_fem.run_fem_alldeg(new_run=new_run)
    
    print(f"\n###### PINNs")
    gains_enhanced_fem.run_pinns_alldeg(u_theta,new_run=new_run)
    
    if "Add" in global_enmethods:
        print(f"\n###### Add")
        gains_enhanced_fem.run_corr_alldeg(u_theta,new_run=new_run)
        
        compare_gains_methods.create_dataframes_alldeg_allM()
        compare_gains_methods.save_stats_alldeg_allM()
    
    for method in global_enmethods:
        if "Mult" in method:
            print(f"\n###### {method}")
            
            tab_M = tab_enmethods[method]

            impose_bc=True
            if method[-1] == "W":
                impose_bc=False
            
            gains_enhanced_fem.run_mult_alldeg_allM(u_theta,tab_M,impose_bc=impose_bc,new_run=new_run)

            compare_gains_methods.create_dataframes_alldeg_allM(tab_M)
            compare_gains_methods.save_stats_alldeg_allM(tab_M)


def run_tests():    
    tests_config = select_tests(answer=False)

    print("\n#################")
    print("### Run tests ###")
    print("#################\n")

    # Select testcase
    config = tests_config["config"]
    dimension = config["dimension"]
    testcase = config["testcase"]
    version = config["version"]  
    
    print("\n############################## Load network ##############################\n")

    # Import the problem, the pde and the network
    netmodule = importlib.import_module(f"enrichedfem.networks.test_{dimension}D.test_{testcase}.test_{testcase}_v{version}")
    pbmodule = importlib.import_module(f"enrichedfem.problem.problem_{dimension}D")
    
    problem = getattr(pbmodule, f"TestCase{testcase}")(version=version)
    print(problem)
    pde = getattr(netmodule,f"Poisson_{dimension}D")()
    _,u_theta = getattr(netmodule,f"Run_laplacian{dimension}D")(pde)
    
    if "EE" in tests_config:
        
        eeconfig = tests_config["EE"]
        run_error_estimates(eeconfig,problem,u_theta)
    
    if "gains" in tests_config:
        
        gainsconfig = tests_config["gains"]
        run_gains(gainsconfig,problem,u_theta)
    
if __name__ == "__main__":
    run_tests()