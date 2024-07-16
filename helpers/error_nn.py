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
import logging
import numpy as np
import tensorflow as tf

from base.nn import NN
from helpers.nn_parametrization import get_param_as_float, has_param, load_and_store_optional_nn_params, load_and_store_optional_training_params, load_training_params, load_nn_params
from helpers.input_generator import generate_collocation_points
from helpers.csv_helpers import export_csv, import_csv
from helpers.globals import get_prefix, set_activation_function, get_optimizer
from helpers.run import eval_pinn

APPL_PATH = None
LOADWEIGHTS = True
CFG_ERROR_NN = "config_error_nn.json"
CFG_ERROR_TRAINING = "config_error_training.json"

class ENN(NN):
    """
    This class defines a purely data driven error neural network
    """
    def __init__(self, layers: list, lb: np.ndarray, ub: np.ndarray) -> None:
        """
        Initializes network in parent class and overwrites loss with weighted loss

        Underestimation penalty is initialized to 1 (underestimation is penalized as strong as underestimation)
        """
        super().__init__(layers, lb, ub)
        self.loss_object = self.loss
        self.w_underestimation = 1.0

    def loss(self, y, y_pred):
        """
        Loss function with the optional extension to penalize underestimation more than overestimation of the error
        """
        y = tf.reshape(y, y_pred.shape)
        weights = tf.cast(tf.math.less(y_pred - y, tf.zeros(y.shape, dtype=tf.dtypes.float64)), dtype=tf.dtypes.float64)*np.abs(self.w_underestimation-1) + tf.ones(y.shape, dtype=tf.dtypes.float64)
        L_data = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred)*weights, axis=1))
        
        return L_data

    def set_underestimation_penalty(self, wu):
        self.w_underestimation = wu

def train_error_nn(X_data=None, E_data=None, LOAD_WEIGHTS=True):
    """
    Trains error neural network. In case no X_data and E_data are given, previously generated data is automatically reused. An exception will occur if no such data is available

    :param np.array X_data : input data for error NN
    :param np.array E_data : reference data for error NN
    :param boolean LOAD_WEIGHTS: defines if weights of NN are loaded before training with X_data and E_data
    """
    if(X_data is None and not LOAD_WEIGHTS) or (E_data is None and not LOAD_WEIGHTS):
        logging.error("Invalid combination of data restoring and data provision. - train_error_nn")
        return

    global APPL_PATH, CFG_ERROR_TRAINING, CFG_ERROR_NN

    # Load parameters for training
    epochs, _ = load_training_params(os.path.join(APPL_PATH, CFG_ERROR_TRAINING))
    load_and_store_optional_training_params(os.path.join(APPL_PATH, CFG_ERROR_TRAINING))

    # Load NN parameters
    input_dim, output_dim, N_layer, N_neurons, lb, ub = load_nn_params(os.path.join(APPL_PATH,CFG_ERROR_NN))
    load_and_store_optional_nn_params(os.path.join(APPL_PATH,CFG_ERROR_NN))

    # NN initialization + setting of underestimation penalty
    error_nn = ENN([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)
    if has_param(os.path.join(APPL_PATH, CFG_ERROR_TRAINING), "w_underestimation"):
        error_nn.set_underestimation_penalty(get_param_as_float(os.path.join(APPL_PATH, CFG_ERROR_TRAINING), "w_underestimation"))

    # NN parametrization by stored weights
    weights_path = os.path.join(APPL_PATH, 'output_data', 'error' + get_prefix()+'weights')
    if LOAD_WEIGHTS:
        error_nn.load_weights(weights_path)

    gen_data_path = os.path.join(APPL_PATH, "output_data", 'error' + get_prefix()+'weights' , "gen_data.csv")
    if X_data is None:
        data = import_csv(gen_data_path, False)
        X_data = data[:, :input_dim]
        E_data = data[:, -2:]
    else:
        export_csv(np.concatenate([X_data, E_data], axis=1), gen_data_path)

    error_nn.fit(X_data[:, :input_dim], E_data[:,0]+E_data[:,1], epochs, None, None, optimizer=get_optimizer(),learning_rate=0.1, val_freq=1000, log_freq=1000)
    error_nn.save_weights(os.path.join(weights_path, 'easy_checkpoint.keras'))


def load_function_error_training(filename= None):
    """
    Loads or generates data for error function training

    :param string filename: If filename is given, data is loaded from this file, otherwise, new files will be generated according to parameters set in config files.
    """
    if filename is not None:
        vals = import_csv(filename)
        n_phys = vals.shape[0]
        X_data = vals
    else:
        global APPL_PATH, CFG_ERROR_NN, CFG_ERROR_TRAINING
        _, _, _, _, lb, ub = load_nn_params(os.path.join(APPL_PATH, CFG_ERROR_NN))
        _, n_phys = load_training_params(os.path.join(APPL_PATH, CFG_ERROR_TRAINING))
        X_data = generate_collocation_points(n_phys, lb, ub)

    return n_phys, X_data, None

def train_error_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None):
    """
    Callout to be inserted as post_eval_callout in a run of the initial neural network. 

    :param np.array X_data : input data for error NN
    :param np.array E_data : reference data for error NN
    other parameters are included to satisfy API requirements of a post eval callout but taken to no use
    """
    global APPL_PATH, CFG_ERROR_NN, CFG_ERROR_TRAINING
    train_error_nn(X_data, E_pred, False)

def generate_error_training_data_and_train(appl_path, create_fun, epsilon, inputfile=None, load_weights=True):
    """
    This function dispatches calls to train the error neural net either directly to training the neural net based on previous generated data or calls the original network first to be evaluated to generate input data for the data-driven error neural network.

    :param string appl_path: path to application under investigation
    :param function create_fun: factory function which creates the application version of the PINN
    :param float epsilon: contribution the numerical integration may give to the a posteriori error estimator
    :param string input_file: In case, the PINN shall be evaluated, so if load_weights=false, the file is used as input for running the PINN
    :param boolean load_weights: Defines if previously generated weights and data shall be used to train the error neural network (true) or if new data shall be generated (false)
    """
    global APPL_PATH, LOADWEIGHTS
    APPL_PATH = appl_path
    LOADWEIGHTS = load_weights

    if load_weights:
        train_error_nn()
    else:
        eval_pinn(create_fun, load_function_error_training, inputfile, appl_path, epsilon, False, callout=train_error_callout)

def eval_error_nn(appl_path, error_nn_config, X_data):
    """
    Function to evaluate the error neural network for X_data

    :param string appl_path: Path to the main directory of the PINN under investigation
    :param string error_nn_config: filename of the NN configuration for the error NN
    :param np.array X_data: input data for which the error NN shall be evaluated.
    """
    input_dim, output_dim, N_layer, N_neurons, lb, ub = load_nn_params(os.path.join(appl_path, error_nn_config))
           
    # NN initialization
    logging.info("Initialize and parametrize error NN...")
    error_nn = ENN([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)

    # NN parametrization by stored weights
    weights_path = os.path.join(appl_path, 'output_data', 'error' + get_prefix()+'weights')
    error_nn.load_weights(weights_path)
    
    logging.info("Evaluate error NN...")
    if X_data.ndim == 1:
        X_data = np.reshape(X_data, (X_data.shape[0], 1))

    E_pred = error_nn.model.predict(X_data[:,:input_dim])
    return E_pred

