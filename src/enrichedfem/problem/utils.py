import os
import numpy as np
import matplotlib.pyplot as plt

def create_tree(path):
    """Creates a directory tree from a given path.

    This function takes a path string and creates all necessary directories
    in the tree. It handles both absolute and relative paths.

    Args:
        path (str): The path string representing the directory tree to create.
    """
    path_split = path.split("/")
    if path[0]=="/":
        path_split = path_split[1:]
        start = "/"
    else:
        start = ""
    for i in range(1,len(path_split)+1):
        subdir = "/".join(path_split[:i])
        if not os.path.isdir(start+subdir):
            os.mkdir(start+subdir)
            
def get_random_param(i, parameter_domain):
    """Generates a random parameter set within a specified domain.

    This function generates a random parameter set based on the provided
    parameter domain and an index 'i', which is used to seed the random
    number generator. The generated parameters are rounded to two decimal places.

    Args:
        i (int): An index used to seed the random number generator.
        parameter_domain (list): A list of tuples, where each tuple defines
            the lower and upper bounds for a parameter.

    Returns:
        list or numpy.ndarray: A list or array of random parameters.
    """
    # pick 1 random parameter at a given index
    dim_params = len(parameter_domain)
    np.random.seed(0)
    for j in range(i):
        param = []
        for k in range(dim_params):
            param.append(np.random.uniform(parameter_domain[k][0], parameter_domain[k][1]))
    param = np.round(param, 2)
    return param

def get_random_params(n_params, parameter_domain):
    """Generates multiple random parameter sets within a specified domain.

    This function generates 'n_params' random parameter sets, where each set
    is drawn from a uniform distribution within the provided parameter domain.
    The random number generator is seeded for reproducibility.

    Args:
        n_params (int): The number of parameter sets to generate.
        parameter_domain (list): A list of tuples, where each tuple defines
            the lower and upper bounds for a parameter.

    Returns:
        numpy.ndarray: A 2D array of random parameters, where each row
            represents a parameter set.
    """
    # pick n random parameters
    dim_params = len(parameter_domain)

    np.random.seed(0)
    params = []
    for i in range(dim_params):
        parami = np.random.uniform(parameter_domain[i][0], parameter_domain[i][1], n_params)
        params.append(parami)
    params = np.array(params).T
    return params

def select_param(problem, param_num):
    """Selects a parameter set for a given problem.

    This function selects a parameter set based on the provided problem and
    parameter number (`param_num`). It prioritizes using predefined parameter
    sets from `problem.set_params` if available. Otherwise, it generates a
    random parameter set within the problem's parameter domain. A specific
    parameter set is hardcoded for `problem.testcase == 3` and `param_num == 3`.

    Args:
        problem: The problem object, which should have either a `set_params`
            attribute or a `parameter_domain` attribute.
        param_num (int): The index of the parameter set to select (1-based).

    Returns:
        list or numpy.ndarray: The selected parameter set.
    """
    try:
        set_params = problem.set_params
    except:
        set_params = None
        
    if set_params is not None:
        assert 1 <= param_num <= len(set_params), f"param_num={param_num} should be less than {len(set_params)}"
        param = set_params[param_num-1]
    else:
        parameter_domain = problem.parameter_domain
        param = get_random_param(param_num,parameter_domain)
        if problem.testcase == 3 and param_num == 3:
            print("ATTENTION : paramètre 3 choisi à la main")
            param = [0.46,0.52,0.12,0.05]
        
    return param

def compute_slope(i, tab_nb_vert, tab_err):
    """Computes the slope between two points on a log-log plot.

    This function calculates the slope of the line segment connecting two
    points on a log-log plot, given by the i-th and (i-1)-th elements of
    'tab_nb_vert' and 'tab_err'. It also plots this segment and returns
    the midpoint of the segment.

    Args:
        i (int): The index of the point in 'tab_nb_vert' and 'tab_err'.
        tab_nb_vert (list or numpy.ndarray): A list or array of x-coordinates.
        tab_err (list or numpy.ndarray): A list or array of y-coordinates.

    Returns:
        tuple: A tuple containing the computed slope and the midpoint of the
            segment on the log-log plot.
    """
    start = [tab_nb_vert[i],tab_err[i]]
    end = [tab_nb_vert[i-1],tab_err[i-1]]
    third = [end[0],start[1]]

    tri_x = [end[0], third[0], start[0], end[0]]
    tri_y = [end[1], third[1], start[1], end[1]
    ]
    plt.plot(tri_x, tri_y, "k--", linewidth=0.5)

    slope = -(np.log(start[1])-np.log(end[1]))/(np.log(start[0])-np.log(end[0]))
    slope = slope.round(2)
    
    vert_mid = [(end[0]+third[0])/2., (end[1]+third[1])/2.]
    
    return slope,vert_mid