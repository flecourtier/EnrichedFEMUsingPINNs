import json
import numpy as np
from utils import get_str,read_config,binary_question
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir[-1] != "/":
    current_dir += "/"
    
def choice_dimension():
    return [1, 2]

def choice_testcase(dimension):
    if dimension == 1:
        pb = ["1D Poisson problem with Dirichlet BC",
                "1D general elliptic system and convection-dominated regime with Dirichlet BC"]
    elif dimension == 2:
        pb = ["2D Poisson problem with low frequency in a square domain with Dirichlet BC",
                "2D Poisson problem with high frequency in a square domain with Dirichlet BC",
                "2D anisotropic elliptic problem on a square domain with Dirichlet BC",
                "2D Poisson problem on an annulus with mixed boundary conditions"]
    choice = np.arange(1,len(pb)+1)
    return list(choice),pb

def choice_version_1D(testcase):
    if testcase == 1:
        v = ["PINNs network by imposing exact BC",
                "Data network by imposing exact BC"]
    else:
        v = ["PINNs network by imposing exact BC"]
    choice = np.arange(1,len(v)+1)
    return list(choice),v

def choice_version_2D(testcase):
    if testcase == 1:
        v = ["PINNs network by imposing exact BC (classical training)",
                "PINNs network by imposing exact BC (Sobolev training)",
                "PINNs network by considering a BC loss (classical training)"]
    elif testcase == 2:
        v = ["PINNs network by imposing exact BC (MLP w/ FF)"]
    elif testcase == 3:
        v = ["PINNs network by imposing exact BC"]
    elif testcase == 4:
        v = ["PINNs network by imposing exact BC"]
    choice = np.arange(1,len(v)+1)
    return list(choice),v

def choice_version(dimension,testcase):
    if dimension == 1:
        return choice_version_1D(testcase)
    elif dimension == 2:
        return choice_version_2D(testcase)
    


def check_config(config_file):
    config = read_config(config_file,current_dir)
    
    # Check the dimension
    assert config["dimension"] in choice_dimension(), f"Dimension must be : {choice_dimension()}"
    
    # Check the testcase
    possibility,_ = choice_testcase(config["dimension"])
    assert config["testcase"] in possibility, f"Testcase must be : {possibility}"
    
    # Check the version
    possibility,_ = choice_version(config["dimension"],config["testcase"])
    assert config["version"] in possibility, f"Version must be : {possibility}"
    
    print(f"### Configuration file {config_file} is valid")
    
    return config

def select_testcase(answer=None):
    if answer is None:
        answer = binary_question(input("-> Would you like to create a configuration file? [Y/N]  "))
    
    if answer:
        print("\n### Creation of the configuration file ###")
    
        # Question 1: Ask for the dimension
        possibility_dim = choice_dimension()
        print(f"\n## Dimension : The available dimensions are: {possibility_dim}")
        dimension = int(input(f"-> Which dimension would you like to consider? "))
        assert dimension in possibility_dim, f"Dimension must be : {possibility_dim}"
    
        # Question 2: Ask for the test case
        possibility_testcase,pb_testcase = choice_testcase(dimension)
        print(f"\n## Testcase : The available testcases are: {possibility_testcase}")
        print(get_str(pb_testcase))
        if len(possibility_testcase) == 1:
            test_case = 1
        else:
            test_case = int(input(f"-> Which testcase would you like to run? "))
        assert test_case in possibility_testcase, f"Testcase must be : {possibility_testcase}"

        # Question 3: Ask for the version
        possibility_version,pb_version = choice_version(dimension,test_case)
        print(f"\n## Version : The available version are: {possibility_version}")
        print(get_str(pb_version))
        if len(possibility_version) == 1:
            version = 1
        else:
            version = int(input(f"-> Which version would you like to run? "))
        assert version in possibility_version, f"Version must be : {possibility_version}"
    
        # Store the data in a dictionary
        config = {
            "dimension" : dimension,
            "testcase": test_case,
            "version": version
        }
    else:
        print("\n### Selection of the configuration file ###")

    config_file = input("-> Enter the name of the configuration file [Default : testcase]: ")
    if config_file == "":
        config_file = "testcase"
    if config_file[-5:] != ".json":
        config_file = config_file + ".json"

    # Write the dictionary to a json file
    if answer:
        with open(current_dir+config_file, "w") as json_file:
            json.dump(config, json_file, indent=4)

        print(f"### Configuration has been saved to {config_file}")
    else:    
        print(f"### Configuration file selected: {config_file}")
    
    config = check_config(config_file)    
    print(config)
        
    print("\n### The configuration file has the following parameters:")
    dimension = config["dimension"]
    test_case = config["testcase"]
    version = config["version"]
    
    print(f"## Dimension: {dimension}D")
    _,pb_testcase = choice_testcase(dimension)
    print(f"## Testcase: {config['testcase']} ({pb_testcase[test_case-1]})")
    _,pb_version = choice_version(dimension,test_case)
    print(f"## Version: {config['version']} ({pb_version[version-1]})")
    
    return config
    

if __name__ == "__main__":
    # Call the function
    config = select_testcase()
