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
import os
import json
import sys

import tensorflow as tf
import numpy as np

from helpers.nn_parametrization import get_param_as_float, has_param, load_and_store_optional_nn_params, load_ee_params, load_nn_params
from helpers.globals import get_prefix
from helpers.csv_helpers import export_csv, import_csv
from helpers.norms import norm

def print_kpis(K, mu, L, delta_mean, appl_path, r0=None, M=None):
    # export data to kpi json
    data = {}
    data['K'] = K
    data['mu'] = mu
    data['L_f'] = L
    data['delta_mean'] = delta_mean
    if r0 is not None:
        data['r0'] = r0
    if M is not None:
        data['M'] = M

    out_file = open( os.path.join(appl_path, 'output_data', get_prefix()+'kpis.json'), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()

def update_kpis(appl_path, update_var, update_var_value):
    K, mu, Lf, deltamean, M = load_ee_params(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"))
    r0 = None
    M = None
    if has_param(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"), "r0"):
        r0 = get_param_as_float(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"), "r0")
    if has_param(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"), "M"):
        M = get_param_as_float(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"), "M")
        
    data = {}
    data['K'] = K
    data['mu'] = mu
    data['L_f'] = Lf
    data['delta_mean'] = deltamean
    if r0 is not None:
        data['r0'] = r0
    if M is not None:
        data['M'] = M

    data[update_var] = update_var_value

    out_file = open( os.path.join(appl_path, 'output_data', get_prefix()+'kpis.json'), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()

def extract_kpis(create_fun, appl_path, mu_factor, post_extract_callout, ae , load_fun=None, input_file=None):
    """
    Extract key performance indicators for a posteriori error estimation. KPIs are then stored in a KPI file to be persistently accessible for other evaluation steps 

    :param function create_fun: factory function which creates PINN 
    :param string appl_path: path to application main directory. Relative to this, output data will be stored
    :param float mu_factor: smoothening parameter for creating delta from deviation R relative to deltamean
    :param function post_extract_callout: callout to be called after parameter extraction
    """
    # set params for determinism
    # tf.config.threading.set_inter_op_parallelism_threads(1) 
    # tf.config.threading.set_intra_op_parallelism_threads(1) 
    # tf.config.set_soft_device_placement(True)

    # Load parameters
    input_dim, output_dim, N_layer, N_neurons, lb, ub = load_nn_params(os.path.join(appl_path, 'config_nn.json'))
    load_and_store_optional_nn_params(os.path.join(appl_path,'config_nn.json'))

    # PINN initialization
    pinn = create_fun([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)

    # PINN parametrization by stored weights
    weights_path = os.path.join(appl_path, 'output_data', get_prefix() +'weights')
    pinn.load_weights(weights_path)
    X_phys = tf.convert_to_tensor(import_csv(os.path.join(weights_path, "collocation_points.csv")))

    # determine mu
    mu, delta_mean, deltaabs = get_mu_deltamean(pinn, X_phys, ub, lb, mu_factor, ae=='domain')
    
    # determine K 
    K  = get_K(pinn, X_phys, mu, ae=='domain')

    # determine r0
    r0 = get_r0(pinn, load_fun, input_file)

    # detrmine boudnary error max norm
    _ = get_boundaryerrmaxnorm(pinn, appl_path)
    
    # print results 
    print_kpis(K, delta_mean.numpy()*mu_factor, pinn.get_Lf(), delta_mean.numpy(), appl_path, r0, M=pinn.get_M())
    
    if post_extract_callout is not None:
        post_extract_callout(X_phys, deltaabs**0.5,os.path.join(appl_path, 'output_data') )

def get_mu_deltamean(pinn, X_phys, ub, lb, mu_factor, isdomain):
    """
    Determines the average deviation of the NN representing the PDE

    param pinn pinn: Instance of class pinn for the current problem
    param tf.tensor X_phys: collocation points used
    param np.array ub: upper bound of considered input space
    param np.array lb: lower bound of considered input space
    param double mu_factor: scaling factor of smoothening addition
    param boolean isdomain: if domain computation is performed
    """
    logging.info("Determine mu and delta_mean")

    # determine the deviation of the PDE on all the collocation points
    val = tf.squeeze(pinn.f_model(X_phys)**2)

    # reduce dimensions if necessary
    if tf.rank(val) >1:
        val =  tf.reduce_sum( val , axis=1)
    
    # determine mean
    delta_mean = tf.reduce_mean(val)**0.5

    # scale by domain size if function based 
    if isdomain:
        if lb.shape[0] > 2:
            delta_mean = delta_mean*np.multiply((ub-lb)[1:])
        else: 
            delta_mean = delta_mean*((ub-lb)[1])
    
    #### WHYYYYY HERE????
    mu = tf.cast(tf.fill((X_phys.shape[0],), delta_mean*mu_factor) , 'float64')

    return mu, delta_mean, val

def get_K(pinn, X_phys, mu, isdomain):
    """
    Determines the key parameter K for trapezoidal rule

    param pinn pinn: Instance of class pinn for the current problem
    param tf.tensor X_phys: collocation points used
    param tf.tensor mu: array of smoothing add-ons
    """

    # divide data into time and not time to only watch time variables
    t = X_phys[:,0:1]
    nott = X_phys[:, 1:X_phys.shape[1]]

    # determine K
    logging.info("Determine K")
    with tf.GradientTape() as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            delta = tf.square(pinn.f_model(tf.concat([t, nott], axis=1)))
            if tf.rank(delta).numpy() >1:
                delta = tf.reduce_sum(delta, axis=1)
            ## ATTENTION: THIS STEP IS NOT COMPATIBLE FOR ANY NORM NOT RELYING ON SQUARING; INTEGRATING AND TAKING THE SQUAREROOT.
            loss = tf.multiply(tf.math.exp(-pinn.get_Lf()*t),tf.reshape((delta + mu**2)**(0.5), t.shape))
        dtloss = tape1.gradient(loss, t)
    dtdtloss = tape2.gradient(dtloss, t)

    # return
    return tf.reduce_max(tf.abs(dtdtloss)).numpy()

def get_r0(pinn, load_fun, input_file):
    """
    Determines the deviations of the initial conditions 

    param pinn pinn: Instance of class pinn for the current problem
    param function load_fun: collocation points used
    param string input_file: input file for initial data
    """
    # assert that the input parameters are correct
    if load_fun is None or input_file is None:
        logging.error("The input of get_r0 may never be None")
        sys.exit()

    logging.info("Determine r0")

    # load and compute data
    _, Xdata, Ydata = load_fun(input_file)
    Ypred = pinn.model(Xdata)

    # determine reduced input dimensions
    out_dim_new = pinn.output_dim
    if pinn.has_output_not_completely_determined_by_PDE():
        out_dim_new = int(pinn.get_output_dim_completely_determined_by_PDE())

    # compute norm
    I1, I2, E = norm(Ypred - Ydata, Xdata[:,1:3], out_dim_new)
    r0 = I1 + E

    # return
    return r0

def get_boundaryerrmaxnorm(pinn, appl_path):
    boundary_error_max_norm = None
    if not pinn.has_hard_constraints():
        boundary_error_max_norm = np.sqrt(tf.reduce_sum(tf.square(pinn.space_model()), axis=1).numpy())
        times = pinn.t_space[:,0]
        times_unique = np.sort(np.unique(pinn.t_space[:,0]))
        bound_err_max_arr = np.zeros(times_unique.shape)

        for index in range(0, times_unique.shape[0]):
            relevant_times = np.argwhere( times <= times_unique[index])
            bound_err_max_arr[index] = np.sqrt(tf.reduce_max(boundary_error_max_norm[relevant_times]))

        export_csv(np.concatenate([np.reshape(times_unique, (times_unique.shape[0],1)), np.reshape(bound_err_max_arr, 
            (bound_err_max_arr.shape[0],1))], axis=1), os.path.join(appl_path, 'output_data', 'bc_err_max_norm.csv'), columnheaders=np.array(["t", "bcerr"]))

    return boundary_error_max_norm
