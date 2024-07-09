###################################################################################################
# Copyright (c) 2022 Birgit Hillebrecht
#
# To cite this code in publications, please use
#       B. Hillebrecht and B. Unger : "Certified machine learning: Rigorous a posteriori error bounds for PDE defined PINNs", arxiV preprint available
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################

import json
import logging
import numpy as np

from helpers.globals import set_activation_function, set_learning_rate, set_storage_frequency, set_log_frequency, set_optimizer, set_validation_frequency, set_w_adaptivity, set_w_adaptivity_factor, set_w_data

def get_current_config(filepath, step=-1):
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()  
    
    returndata = data
    if step != -1:
        for dataentry in data:
            if dataentry['step'] == step:
                returndata = dataentry

    return returndata

def load_training_params(filepath, step=-1):
    """
    Loads parameters to configure the training from the json file provided via param filepath

    returns
    - epochs: number of epochs used for training
    - n_phys: number of collocation points for the PINN
    """

    data = get_current_config(filepath, step)
    epochs = 0
    n_phys = 0

    try :
        epochs = int(data['epochs'])
    except :
        logging.error("Cannot load epochs from configuration file: Either the configuration file is invalid or it is required to indicate the step (-s) option.")
        exit(1)
    if epochs <= 0:
        logging.error("n_phys may not be negative.")
        exit(1) 

    try :
        n_phys = int(data['n_phys'])
    except :
        logging.error("Cannot load n_phys from configuration file: Either the configuration file is invalid or it is required to indicate the step (-s) option.")
        exit(1)
    if n_phys <= 0:
        logging.error("n_phys may not be negative.")
        exit(1)

    return epochs, n_phys
    
def load_and_store_optional_training_params(filepath, step=-1):

    data = get_current_config(filepath, step)

    if has_param(data, "optimizer"):
        set_optimizer(data["optimizer"])

    if has_param(data, "learning_rate"):
        set_learning_rate(data["learning_rate"])

    if has_param(data, "validation_frequency"):
        set_validation_frequency(data['validation_frequency'])
    
    if has_param(data, "log_frequency"):
        set_log_frequency(data["log_frequency"])

    if has_param(data, "storage_frequency"):
        set_storage_frequency(data["storage_frequency"])
    else: 
        set_storage_frequency(0)

    if has_param(data, "w_data"):
        set_w_data(data["w_data"])

    if has_param(data, "w_adapt"):
        set_w_adaptivity(data["w_adapt"])

    if has_param(data, "w_adapt_alpha_init"):
        set_w_adaptivity_factor(data["w_adapt_alpha_init"])
    return

def load_nn_params(filepath):
    """
    Loads parameters for the neural network from the json file provided via param filepath.

    returns 
    - input_dim: number of input nodes
    - output_dim: number of output nodes
    - num_layers: number of layers
    - num_neurons: number of nodes per layer
    - lower_bound: lower bounds on input node values
    - upper_bound: upper bounds on input node values
    - af (None): if set in json, the value for the activation function
    """

    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    lb = np.array(data['lower_bound'])
    ub = np.array(data['upper_bound'])

    return int(data['input_dim']), int(data['output_dim']), int(data['num_layers']), int(data['num_neurons']), lb.astype(np.float64), ub.astype(np.float64)

def load_and_store_optional_nn_params(filepath, step=-1):

    data = get_current_config(filepath, step)
    if has_param(data, "activation_function"):
        set_activation_function(get_param_as_string(filepath, "activation_function"))
    
    return

def load_ee_params(filepath):
    """
    Loads parameters for a posteriori error estimation for the PINN from the json file provided via param filepath.

    returns 
    - K: as used for trapezoidal rule
    - mu: smoothing parameter for delta function
    - L_f: Lipschitz constant or spectral abscissa
    - delta_mean: average deviation of approximated ODE/PDE from target ODE/PDE
    """
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    M = 1.0
    if has_param(filepath,  "M"):
        M = get_param_as_float(filepath, "M")

    return float(data['K']), float(data['mu']), float(data['L_f']), float(data['delta_mean']), M

def has_param(data, param_name):
    """
    Checks if keyword param_name exists in json file

    :param string data: config data struct
    :param string param_name: keyword used for parameter in json file.
    """
    try: 
        data[param_name] 
    except KeyError as e:
        return False

    return True

def get_param_as_float(filepath, param_name):
    """
    Extracts parameter as float from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return float( data[param_name] )

def get_param_as_int(filepath, param_name):
    """
    Extracts parameter as integer from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """    
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return int( data[param_name] )

def get_param_as_array(filepath, param_name):
    """
    Extracts parameter as array from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """    
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return np.array( data[param_name] )

def get_param_as_string(filepath, param_name):
    """
    Extracts parameter as string from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """    
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return str( data[param_name] )

def get_param_as_boolean(filepath, param_name):
    """
    Extracts parameter as boolean from json file. Does not check existance, misuse may lead to exceptions.
    Call has_param first.

    :param string filepath: path to json file
    :param string param_name: keyword used for parameter in json file.
    """        
    with open(filepath, "r") as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()

    return bool( (data[param_name] == "True") or (data[param_name] == "true") ) 