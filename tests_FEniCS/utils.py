import os
import json

def binary_question(answer):
    answer = answer.upper()
    assert answer in ["Y", "N"], "Answer must be Y or N"
    
    if answer == "Y":
        return True
    return False

def qcm(answer,possibility,error_str):
    print(answer)
    if answer == 'A' or answer == 'a':
        tab = possibility
    else:
        tab = list(map(int, answer.split(",")))
        tab_assert(tab,possibility,error_str)
    return tab
        

def tab_assert(tab,possibility,error_str):
    for t in tab:
        assert t in possibility, error_str
        
    
def read_config(configfile,current_dir=""):
    assert os.path.exists(current_dir+configfile), f"File {configfile} does not exist"
    
    # Read the config file
    with open(current_dir+configfile, "r") as json_file:
        config = json.load(json_file)

    return config

def get_str(pb):
    infos = ""
    for i in range(len(pb)):
        infos += f"{i+1}. {pb[i]}"
        if i != len(pb)-1:
            infos += "\n"
    return infos

def print_dict(dic,ind="-"):
    for key, value in dic.items():
        if not isinstance(value,dict):
            print(f"{ind}{key} : {value}")
        else:
            print(f"{ind}{key} :")
            print_dict(value,ind+"-")