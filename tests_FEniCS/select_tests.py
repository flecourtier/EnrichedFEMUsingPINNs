import json
import numpy as np
import os
from utils import binary_question,qcm,get_str,read_config,tab_assert,print_dict
from select_testcase import select_testcase
from enrichedfem.problem import problem_1D,problem_2D

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir[-1] != "/":
    current_dir += "/"

def choice_degree(config):
    if config['dimension'] == 1:
        degree = [1]
    else:
        degree = [1,2,3]
    return degree

def choice_methods(config):
    if config['dimension'] == 1:
        if config['testcase'] == 1:
            if config['version'] == 1:
                methods = {"FEM":None,"Add":None,"Mult":[3.0,100.0]}
        else:
            methods = {"FEM":None,"Add":None,"Mult":[0.0],"MultW":[0.0]}
    else:
        methods = {"FEM":None,"Add":None}
    return methods

def choice_nparams(config):
    n_params = 50
    if config['dimension'] == 1 and config['testcase'] == 1:
        n_params = 100
    return n_params

def ask_error_estimates(tests_config):    
    config = tests_config["config"] 
    possibility_degree = choice_degree(config)
    pb_methods = choice_methods(config)
    
    run_error_estimates = binary_question(input(f"-> Would you like to run the error estimates? [Y/N]  "))
        
    if run_error_estimates:
        # Degree
        print(f"\n## Degree : The available degrees are: {possibility_degree}")
        tab_degree = qcm(input(f"-> Which degrees would you like to consider [A - All] ? "),possibility_degree,f"Degree must be : {possibility_degree}")
        
        # Methods
        possibility_methods = list(np.arange(1,len(pb_methods)+1))
        print(f"\n## Methods : The available methods are: {possibility_methods}")
        global_methods = list(pb_methods.keys())
        print(get_str(global_methods))
        tab_indmethods = qcm(input(f"-> Which methods would you like to consider [A - All] ? "),possibility_methods,f"Methods must be : {possibility_methods}") 
        tab_indmethods = list(map(lambda x: x - 1, tab_indmethods))
        tab_methods = list(np.array(global_methods)[tab_indmethods])
        
        dict_methods = {}
        for method in tab_methods:
            if pb_methods[method]:
                print(f"\n# {method} :")
                print(get_str(pb_methods[method]))
                tab_sub = qcm(input(f"-> Which sub-methods would you like to consider for {method} [A - All] ? "),pb_methods[method],f"Sub-methods must be : {pb_methods[method]}")
                dict_methods[method] = tab_sub
            else:
                dict_methods[method] = None
        
        # Parameters
        print(f"\n## Parameters :")
        new_generation = True
        tab_param_num = []
        tab_params = []
        
        random = binary_question(input(f"-> Would you like to randomly draw parameters? [Y/N]  "))
        while new_generation:
            if random:
                tab_param_num.append(int(input(f"-> Give an index of parameters : ")))
            else:
                class_name = f"TestCase{config['testcase']}"
                if config['dimension'] == 1:
                    cls = getattr(problem_1D, class_name)  
                else:
                    cls = getattr(problem_2D, class_name)
                problem = cls()
                nb_parameters = problem.nb_parameters
                parameter_domain = problem.parameter_domain
                
                print("# The parameters domains are:")
                for i in range(nb_parameters):
                    print(f"-> Parameter {i+1} : {parameter_domain[i]}")
                
                params = input(f"-> Give the parameters [p1,p2,...] : ")
                params = list(map(float, params.split(",")))
                for i in range(nb_parameters):
                    assert parameter_domain[i][0] <= params[i] <= parameter_domain[i][1], f"Parameter {i+1} must be in {parameter_domain[i]}"
                tab_params.append(params)
                    
            new_generation = binary_question(input(f"-> Would you like to generate a new parameter? [Y/N]  "))
            
        tests_config["EE"] = {"degree":tab_degree,"methods":dict_methods}
        if random:
            tests_config["EE"]["param_num"] = tab_param_num
        else:
            tests_config["EE"]["params"] = tab_params
            
    return tests_config

def ask_gains(tests_config):   
    config = tests_config["config"] 
    possibility_degree = choice_degree(config)
    pb_methods = choice_methods(config)
    # Remove FEM from the methods
    del pb_methods["FEM"]
    n_params = choice_nparams(config)
    
    run_gains = binary_question(input(f"-> Would you like to run the gains of enriched FEM? [Y/N]  "))
    
    if run_gains:
        # Degree
        print(f"\n## Degree : The available degrees are: {possibility_degree}")
        tab_degree = qcm(input(f"-> Which degrees would you like to consider [A - All] ? "),possibility_degree,f"Degree must be : {possibility_degree}")
        
        # Methods
        possibility_methods = list(np.arange(1,len(pb_methods)+1))
        print(f"\n## Methods : The available methods are: {possibility_methods}")
        # print(get_str(pb_methods))
        # tab_indmethods = qcm(input(f"-> Which methods would you like to consider [A - All] ? "),possibility_methods,f"Methods must be : {possibility_methods}") 
        # tab_indmethods = list(map(lambda x: x - 1, tab_indmethods))
        # tab_methods = list(np.array(pb_methods)[tab_indmethods])
        
        global_methods = list(pb_methods.keys())
        print(get_str(global_methods))
        tab_indmethods = qcm(input(f"-> Which methods would you like to consider [A - All] ? "),possibility_methods,f"Methods must be : {possibility_methods}") 
        tab_indmethods = list(map(lambda x: x - 1, tab_indmethods))
        tab_methods = list(np.array(global_methods)[tab_indmethods])
        
        dict_methods = {}
        for method in tab_methods:
            if pb_methods[method]:
                print(f"\n# {method} :")
                print(get_str(pb_methods[method]))
                tab_sub = qcm(input(f"-> Which sub-methods would you like to consider for {method} [A - All] ? "),pb_methods[method],f"Sub-methods must be : {pb_methods[method]}")
                dict_methods[method] = tab_sub
            else:
                dict_methods[method] = None
        
        
        # Number of parameters
        print(f"\n## Number of parameters : {n_params}")
        
        tests_config["gains"] = {"degree":tab_degree,"methods":dict_methods,"n_params":n_params}
            
    return tests_config
                

def check_config(config_file):
    tests_config = read_config(config_file,current_dir)
    config = tests_config["config"]
    
    # Check the dimension
    if "EE" in tests_config:
        ee = tests_config["EE"]
        
        possibility = choice_degree(config)
        tab_assert(ee["degree"],possibility,f"Degree must be : {possibility}")
        possibility = choice_methods(config)
        tab_assert(ee["methods"],possibility,f"Methods must be : {possibility}")
        if "params" in ee:
            tab_params = ee["params"]
            class_name = f"TestCase{config['testcase']}"
            if config['dimension'] == 1:
                cls = getattr(problem_1D, class_name)  
            else:
                cls = getattr(problem_2D, class_name)
            problem = cls()
            nb_parameters = problem.nb_parameters
            parameter_domain = problem.parameter_domain
            
            print("# The parameters domains are:")
            for i in range(nb_parameters):
                print(f"-> Parameter {i+1} : {parameter_domain[i]}")
            
            params = input(f"-> Give the parameters [p1,p2,...] : ")
            params = list(map(float, params.split(",")))
            for i in range(nb_parameters):
                assert parameter_domain[i][0] <= params[i] <= parameter_domain[i][1], f"Parameter {i+1} must be in {parameter_domain[i]}"
            tab_params.append(params)
    
    if "gains" in tests_config:
        gains = tests_config["gains"]
        possibility = choice_degree(config)
        tab_assert(ee["degree"],possibility,f"Degree must be : {possibility}")
        possibility = choice_methods(config)
        tab_assert(ee["methods"],possibility,f"Methods must be : {possibility}")
        assert gains["n_params"] == choice_nparams(config), f"Number of parameters must be : {choice_nparams(config)}"
    
    print(f"### Configuration file {config_file} is valid")
    
    return tests_config

def select_tests(answer=None):
    if answer is None:
        answer = binary_question(input("-> Would you like to create a configuration file? [Y/N]  "))
    
    if answer:
        print("###########################")
        print("### Select the testcase ###")
        print("###########################")
        
        config = select_testcase(answer=False)
        tests_config = {}
        tests_config["config"] = config
    
    print("\n########################")
    print("### Select the tests ###")
    print("########################")
    if answer:        
        print("\n### Creation of the configuration file ###")

        # Question 1: Ask for the error_estimates
        if not (config['dimension'] == 1 and config['testcase'] == 1 and config['version'] == 2):
            print("\n### Error estimates")
            tests_config = ask_error_estimates(tests_config)
            
        # Question 2: Ask for the gains
        print("\n### Gains")
        tests_config = ask_gains(tests_config)
        
    else:
        print("\n### Selection of the configuration file ###")

    config_file = input("-> Enter the name of the configuration file [Default : tests]: ")
    if config_file == "":
        config_file = "tests"
    if config_file[-5:] != ".json":
        config_file = config_file + ".json"

    # Write the dictionary to a json file
    if answer:
        with open(current_dir + config_file, "w") as json_file:
            json.dump(tests_config, json_file, indent=4)

        print(f"### Configuration has been saved to {config_file}")
    else:    
        print(f"### Configuration file selected: {config_file}")
    
    tests_config = check_config(config_file)    
    print("\n### The configuration file has the following parameters:")
    print_dict(tests_config)
    
    return tests_config
    

if __name__ == "__main__":
    # Call the function
    tests_config = select_tests()
