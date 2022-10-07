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

import os

from helpers.nn_parametrization import load_and_store_optional_nn_params, load_and_store_optional_training_params, load_training_params, load_nn_params
from helpers.input_generator import generate_collocation_points
from helpers.csv_helpers import export_csv, import_csv
from helpers.globals import get_learning_rate, get_log_frequency, get_optimizer, get_prefix, get_validation_frequency

def train_pinn(create_fun, load_fun, load_param, LOAD_WEIGHTS, STORE, appl_path, post_train_callout):
    """
    Basic function for training a PINN. 

    :param function create_fun: factory function which creates PINN 
    :param function load_fun: function which loads or creates data points based on load_param
    :param (string) load_param: path to input data (type and actual content depend on load_fun implementation)
    :param boolean LOAD_WEIGHTS: loads weights before training (again) from a predefined path e.g. appl_path/output_data/'tanh_weights' for tanh.
    :param boolean STORE: stores weights after training to a predefined path e.g. appl_path/output_data/'tanh_weights' for tanh.
    :param string appl_path: path to application main directory. Relative to this, output data will be stored
    :param function post_train_callout: callout to be called after training    
    """

    # Load training parameters
    epochs, N_phys = load_training_params(os.path.join(appl_path, 'config_training.json'))
    load_and_store_optional_training_params(os.path.join(appl_path, 'config_training.json'))

    # Load NN parameters
    input_dim, output_dim, N_layer, N_neurons, lb, ub = load_nn_params(os.path.join(appl_path,'config_nn.json'))
    load_and_store_optional_nn_params(os.path.join(appl_path,'config_nn.json'))

    # load input data
    _, X_data, Y_data = load_fun(load_param)
    
    # PINN initialization
    pinn = create_fun([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)

    # PINN parametrization by stored weights
    weights_path = os.path.join(appl_path, 'output_data', get_prefix()+'weights')
    if LOAD_WEIGHTS:
        pinn.load_weights(weights_path)
        X_phys = import_csv(os.path.join(weights_path, "collocation_points.csv"))
    else:
        X_phys = generate_collocation_points(N_phys, lb, ub)

    # PINN training
    # Generate training data via LHS
    pinn.set_collocation_points(X_phys)
    pinn.fit(X_data, Y_data, epochs, None, None, 
        optimizer=get_optimizer(), learning_rate=get_learning_rate(), 
        val_freq=get_validation_frequency(), log_freq=get_log_frequency())

    # store weights
    if STORE:
        print("STORE")
        pinn.save_weights(os.path.join(weights_path, 'easy_checkpoint'))
        print(os.path.join(weights_path, 'easy_checkpoint'))
        export_csv(X_phys, os.path.join(weights_path,"collocation_points.csv") ) 

    post_train_callout(pinn, os.path.join(appl_path, 'output_data'))
