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
import os
import logging
import sys

import numpy as np

from helpers.extract import  update_kpis
from helpers.nn_parametrization import get_param_as_float, has_param, load_and_store_optional_nn_params, load_ee_params, load_nn_params
from helpers.globals import get_prefix
from helpers.csv_helpers import export_csv, import_csv
from helpers.integrate import compute_1d_integral
from helpers.error_helpers import ErrorEstimationType
from helpers.norms import norm

def Compute_Number_of_SupportPoints(time, K, E_ML, epsilon, Nlim=10000) :
    """
    Returns the required number of support points for composite trapezoidal rule

    param double time: time at which it shall be evaluated
    param double K: error defining constant for the composite trapezoidal rule
    param double E_ML: expected machine learning error used to scale the acceptible integration error
    param double epsilon: scales the integration error with respect to the expected machine learning error, suitable choice could be 0.1
    param int Nlim: maximum number of supported support intervals by the underlying machine. 
    """
    return min(np.ceil(np.sqrt(K*time**3 / (12*E_ML*epsilon))).astype(int), Nlim)

def Compute_Expected_ML_Error(time, Lf, r0, deltamean) :
    """
    Computes the expected machine learning error for a fixed time 

    param double time: time at which a prediction shall be made
    param double Lf: growth bound of the semigroup 
    param double r0: error on the initial conditions used
    param double deltamean: average deviation of the NN representing the temporal evolution of the PDE
    """
    if Lf== 0:
        return r0 + deltamean*time
    else:
        return np.exp(Lf * time)*(r0 + (1-np.exp(-time*Lf))*deltamean/Lf)

def Compute_Pointwise_Error(X_data, pinn, K, mu, Lf, deltamean, epsilon, ndim) :
    """
    Function to determine error for input data X_data

    :param array X_data: input data for PINN
    :param PINN pinn: PINN under investigation
    :param float K: key parameter for using trapezoidal rule and estimating the number of required subintervals
    :param float mu: smoothening parameter for creating delta from deviation R
    :param float Lf: Lipschitz constant or spectral abscissa of system under investigation
    :param float deltamean: a priori determined average deviation in ODE/PDE
    :param float epsilon: contribution the error of the numerical integration may give to the overall a posteriori error
    :param int ndim: dimensions of input data  
    """
    # initialize variables for error and number of support points
    E_pred = np.zeros((X_data.shape[0], 2)) 
    N_SP = np.repeat(0, X_data.shape[0], axis=0)

    # compute target value and error for all times
    for x_index in range(X_data.shape[0]):
        # get current item
        x_item = np.reshape(X_data[x_index], (1, X_data.shape[1]))   

        # predict value at time 0 and compare to input values to get r0
        t = x_item[0,0]
        x_item[0,0] = 0
        r0 = np.sqrt(np.sum((pinn.predict(x_item)[0] - x_item[0, -ndim:])**2))
        x_item[0,0] = t

        # compute predicted machine learning error and number of required support points
        E_ML = Compute_Expected_ML_Error(t, Lf, r0, deltamean)
        N_SP[x_index] = Compute_Number_of_SupportPoints(t, K, E_ML, epsilon)

        # compute prediction of support points
        T_test = np.transpose(np.array([np.linspace(0,x_item[0,0],2*(N_SP[x_index]+1))]))
        X_test = np.repeat(x_item, T_test.shape[0], axis=0)
        X_test[:,0]  = T_test[:,0]
        _, F_pred = pinn.predict(X_test)
    
        # compute integral for error
        targetfun = (np.sqrt(np.reshape(np.sum(F_pred**2, axis=1),(F_pred.shape[0],1)) + np.full((F_pred.shape[0],1), mu, dtype="float64")**2) * np.exp(-Lf*T_test))
        targetfun = np.sum(targetfun, axis=1 )
        I1, I2, E = compute_1d_integral(targetfun, T_test)
        
        # determine error
        E_pred[x_index, 0] = np.exp(Lf*x_item[0,0])*(r0)
        if x_item[0,0] != 0:
            E_pred[x_index, 1] = np.exp(Lf*x_item[0,0])*(I1 + E)

        # log regularily
        if x_index % 100 == 0:
            logging.info(f'Predicted error for index {x_index}: {E_pred[x_index]}')
    return E_pred, N_SP

def Compute_Function_Error(targettime, X_range, pinn, K, mu, Lf, deltamean, epsilon, r0, ndim, M=1.0, depth=0):
    # initialize return variables
    E_pred = np.zeros(2)
    N_SP = 0
    K_upd = K

    # check that time is in the allowed range
    if targettime < 0 :
        logging.error("Can't compute error for times smaller 0. Time given: t= "+ targettime)

    elif targettime == 0: 
        E_pred[0] = M*(r0)
        E_pred[1] = 0

    else:
        # compute predicted machine learning error and number of required support points
        E_ML = M*Compute_Expected_ML_Error(targettime, Lf, r0, deltamean)
        N_SP = Compute_Number_of_SupportPoints(targettime, K, E_ML, epsilon)

        t = np.linspace(0, targettime, 2*N_SP+1)
        delta_over_time = np.zeros(t.shape)

        # find values for required support points
        for index_currenttime in range(t.shape[0]):
            input = np.concatenate((np.ones((X_range.shape[0],1))*t[index_currenttime], X_range), axis=1)
            # use 2-norm
            f_pred = pinn.f_model(input)
            I1, I2, E = norm(f_pred, X_range)
            
            # compute error and add it to upper limit of delta_over_time
            delta_over_time[index_currenttime] = I1 + E

        # time integration
        targetfun = (np.sqrt(np.power(delta_over_time,2) + np.full((delta_over_time.shape[0]), mu, dtype="float64")**2) * np.exp(-Lf*t))
        I1, I2, E = compute_1d_integral(targetfun, np.reshape(t, [t.shape[0],1]))

        # determine error
        E_pred[0] = M*np.exp(Lf*targettime)*(r0)
        E_pred[1] = M*np.exp(Lf*targettime)*(I1 + E)

        if targettime > 0 and N_SP > 0:
            K_upd = 12 * E * N_SP**2/ ( targettime**3 )

        N_SP_2 = Compute_Number_of_SupportPoints(targettime, K_upd, E_ML, epsilon)

        if N_SP_2 > N_SP and depth < 4:
            E_pred, N_SP, K_upd = Compute_Function_Error(targettime, X_range, pinn, K_upd, mu, Lf, deltamean, epsilon, r0, ndim, M=M, depth=depth+1)

    return E_pred, N_SP, K_upd

def eval_pinn(create_fun, load_fun, input_file, appl_path, epsilon, eet, callout=None):
    """
    Basic function for running a PINN with input data

    :param function create_fun: factory function which creates PINN 
    :param function load_fun: function which loads or creates data points based on load_param
    :param string input_file: path to input data 
    :param string appl_path: path to application main directory. Relative to this, output data will be stored
    :param float epsilon: fraction of error the error introduced by numerical integration may have
    :param string ae: determines if a posteriori error estimation is executed (not value "none") and decides if it is determined either pointwise or domainwise.
    :param function callout: callout to be called after evaluating the PINN    
    """
    # Load  nn parameters
    input_dim, output_dim, N_layer, N_neurons, lb, ub, = load_nn_params(os.path.join(appl_path,'config_nn.json'))
    load_and_store_optional_nn_params(os.path.join(appl_path,'config_nn.json'))

    # load error estimation parameters
    K, mu, Lf, deltamean, M = load_ee_params(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"))
    r0 = None

    if has_param(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"), "r0"):
        r0 = get_param_as_float(os.path.join(appl_path, "output_data", get_prefix() +  "kpis.json"), "r0")
    bce = None

    # load input values
    if input_file is None:
        logging.warning(f'No input file is given. Load_fun called with None')
        _, X_data, Y_data = load_fun(None)
    else:
        _, X_data, Y_data = load_fun(os.path.join(appl_path, "input_data", input_file))

    # PINN initialization + parametrization by stored weights
    pinn = create_fun([input_dim, *N_layer * [N_neurons], output_dim], lb, ub)
    weights_path = os.path.join(appl_path, 'output_data', get_prefix() + 'weights')
    pinn.load_weights(weights_path)

    # check if hard constraints are given and import boundary condition errors over time
    bc_err = None
    if not pinn.has_hard_constraints():
        bc_err = import_csv(os.path.join(appl_path, "output_data", "bc_err_max_norm.csv"), has_headers=True)
        
    # run pinn
    Y_pred = pinn.model(X_data)

    # format export data and column headers
    E_pred, N_SP_pred = None, None
    if eet == ErrorEstimationType.POINTWISE:
        E_pred, N_SP_pred = Compute_Pointwise_Error(X_data, pinn, K, mu, Lf, deltamean, epsilon, output_dim)

    elif eet == ErrorEstimationType.DOMAINWISE:
        if r0 is None:
            logging.error("For domainwise error estimation it is mandatory to have r0 estimated a priori.")
            sys.exit()

        E_pred, N_SP_pred, K_upd = Compute_Function_Error(X_data[0,0], X_data[:,1:], pinn, K, mu, Lf, deltamean, epsilon, r0, ndim=1, M=M)

        STORE_K_UPD = True
        if STORE_K_UPD:
            update_kpis(appl_path, 'K', K_upd)
            try:
                (open(os.path.join(appl_path, "output_data",'run_'+get_prefix()+'values', "K_hist.txt"), "a")) 
            except:
                os.makedirs(os.path.join(appl_path, "output_data",'run_'+get_prefix()+'values'))
                (open(os.path.join(appl_path, "output_data",'run_'+get_prefix()+'values', "K_hist.txt"), "x")) 

            with open(os.path.join(appl_path, "output_data",'run_'+get_prefix()+'values', "K_hist.txt"), "a") as file_object:
                file_object.write(str(K_upd)+"\n")
    
        if bc_err is not None:
            if np.argwhere(bc_err[:,0] <= X_data[0,0]).shape[0] > 0:
                last_known_time_index = np.max(np.argwhere(bc_err[:,0] <= X_data[0,0]))
                if last_known_time_index is not None:
                    bce = bc_err[last_known_time_index, 1]
                else:
                    bce = 0

    tail = None
    if input_file is not None:
        _, tail = os.path.split(input_file)

    # export data to csv to reuse results
    if input_file is not None:
        export_data = np.concatenate([X_data, Y_pred], axis=1)

        colhdsx = np.core.defchararray.add( np.repeat("x", X_data.shape[1], axis=0), np.linspace(0, X_data.shape[1]-1, X_data.shape[1]).astype('int').astype('str'))
        colhdsy =  np.core.defchararray.add( np.repeat("y", Y_pred.shape[1], axis=0), np.linspace(0, Y_pred.shape[1], Y_pred.shape[1]).astype('int').astype('str')) 
        colhds = np.concatenate([colhdsx, colhdsy], axis = 0)
        
        # determine error
        if eet == ErrorEstimationType.POINTWISE:
            export_data = np.concatenate([export_data, E_pred, np.reshape(N_SP_pred, (N_SP_pred.shape[0],1))], axis = 1)
            colhdse = np.core.defchararray.add(np.core.defchararray.add( np.repeat("E_pred[", 2, axis=0), np.linspace(0, 2, 2).astype('int').astype('str')) , np.repeat("]",2, axis=0))
            colhdsn = np.core.defchararray.add(np.core.defchararray.add( np.repeat("N_SP[", 1, axis=0), np.linspace(0, 1, 1).astype('int').astype('str')) , np.repeat("]",1, axis=0))
            colhds = np.concatenate([colhds, colhdse, colhdsn], axis = 0)

        if eet == ErrorEstimationType.DOMAINWISE:
             # export data to  json
            data = {}
            data['E_init'] = E_pred[0]
            data['E_PI'] = E_pred[1]
            if bce is not None:
                data['E_bc'] = pinn.get_ISS_param()*bce

            data['N_SP'] = int(N_SP_pred)

            out_file = open( os.path.join(appl_path, 'output_data', 'run_'+get_prefix()+'values', "run_" + get_prefix() +tail[:-4] + "_error.json"), "w")
            json.dump(data, out_file, indent=4)
            out_file.close()

        export_csv(export_data, os.path.join(appl_path, 'output_data', 'run_'+get_prefix()+'values', "run_" + get_prefix() +tail), columnheaders=colhds)

    # call post_run_callout
    if callout is not None: 
        callout(os.path.join(appl_path, 'output_data', 'run_'+get_prefix()+'figures'), X_data, Y_pred, E_pred, N_SP_pred, Y_data, tail)

