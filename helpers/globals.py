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

import logging
import sys

import tensorflow as tf

#########################################################################################################
## This files keeps track and handles global variables 
#########################################################################################################

PREFIX = 'tanh_'
ACTIVATION_FUNCTION = tf.nn.tanh
OPTIMIZER = 'adam'
LEARNING_RATE=0.1
VALIDATION_FREQUENCY=1000
LOG_FREQUENCY=1000
STORAGE_FREQUENCY=1000
RES_NET_SKIP_LENGTH = 0
W_DATA = 1.0
W_ADAPTIVITY = False
W_ADAPTIVITY_FACTOR = 0.0

def get_activation_function():
    """
    Returns tf.nn activation function
    """
    return ACTIVATION_FUNCTION

def get_prefix() -> str:
    return PREFIX

def set_activation_function(af) -> None:
    """
    Sets activation function for next NN.

    :param string af: activation function by string, supported are tanh, silu, gelu and softmax
    """
    global ACTIVATION_FUNCTION
    global PREFIX
    PREFIX = af + '_'

    if af == 'tanh':
        ACTIVATION_FUNCTION = tf.nn.tanh
    elif af == 'silu':
        ACTIVATION_FUNCTION = tf.nn.silu
    elif af == 'gelu':
        ACTIVATION_FUNCTION = tf.nn.gelu
    elif af == 'softmax':
        ACTIVATION_FUNCTION = tf.nn.softmax
    elif af == 'relu':
        ACTIVATION_FUNCTION = tf.keras.activations.relu
    else:
        logging.error("The activation function given \'" + af + "\' is invalid. Value must be either tanh, silu, gelu, relu or softmax")
        sys.exit()

def set_res_net_skip_length(skip_length) -> None:
    global RES_NET_SKIP_LENGTH
    RES_NET_SKIP_LENGTH = skip_length

def get_res_net_skip_length() :
    global RES_NET_SKIP_LENGTH
    return RES_NET_SKIP_LENGTH

def set_optimizer(o) -> None:
    """
    Sets optimizer for next NN.

    :param string o: optimizer by string, supported are lbfgs and adam
    """
    global OPTIMIZER
    if not (o == 'lbfgs') and not (o =='adam') and not (o == 'lbfgs_tf_probability'):
        logging.error("The optimizer chosen is not supported: \'"+o+"\'. Valid values are lbfgs, lbfgs_tf_probability and adam.")
        sys.exit()

    OPTIMIZER = o

def get_optimizer() -> str:
    global OPTIMIZER
    return OPTIMIZER

def get_validation_frequency() -> int:
    global VALIDATION_FREQUENCY 
    return VALIDATION_FREQUENCY

def set_validation_frequency(f : int):
    global VALIDATION_FREQUENCY
    VALIDATION_FREQUENCY = f
    return

def get_log_frequency() -> int:
    global LOG_FREQUENCY
    return LOG_FREQUENCY

def set_log_frequency(f: int):
    global LOG_FREQUENCY
    LOG_FREQUENCY = f
    return

def get_storage_frequency() -> int:
    global STORAGE_FREQUENCY
    return STORAGE_FREQUENCY

def set_storage_frequency(f: int):
    global STORAGE_FREQUENCY
    STORAGE_FREQUENCY = f
    return

def get_learning_rate() -> float:
    global LEARNING_RATE
    return LEARNING_RATE

def set_learning_rate(rate : float):
    global LEARNING_RATE 
    LEARNING_RATE = rate
    return

def set_w_adaptivity(w: bool):
    global W_ADAPTIVITY
    W_ADAPTIVITY = w
    return

def get_w_adaptivity():
    global W_ADAPTIVITY
    return W_ADAPTIVITY

def set_w_adaptivity_factor(w: float):
    global W_ADAPTIVITY_FACTOR
    W_ADAPTIVITY_FACTOR = w
    return

def get_w_adaptivity_factor():
    global W_ADAPTIVITY_FACTOR
    return W_ADAPTIVITY_FACTOR

def get_w_data():
    global W_DATA
    return W_DATA

def set_w_data(w:float):
    global W_DATA
    W_DATA = w
    return