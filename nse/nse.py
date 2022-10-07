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
import tensorflow as tf
import numpy as np

from base.bc_layer import BoundaryCondition
from base.pinn import PINN

from helpers.csv_helpers import export_csv, import_csv
from helpers.globals import get_prefix
from helpers.l2_norm import compute_l2_norm

from nse.taylor_bc_offsetlayer import Taylor_BC_OffsetLayer

###############################################################################
## Navier-Stokes Equations for PINN
###############################################################################
class NSE(PINN):

    def __init__(self, layers, lb, ub, X_f=None, bcs= None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub, bcs=bcs)
        self.t = None
        self.x = None
        self.y = None

        self.space_weight = 100.0
        self.has_space_restriction = True

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
        """
        Sets collocation points in class to X_f.
        The components correspond to time (t), and space (x, y) in this order.
        """
        self.t = self.tensor(X_f[:,0:1])
        self.x = self.tensor(X_f[:,1:2])
        self.y = self.tensor(X_f[:,2:3])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the NSE

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
            x = self.x
            y = self.y
        else:
            t = self.tensor(X_f[:, 0:1])
            x = self.tensor(X_f[:, 1:2])
            y = self.tensor(X_f[:, 2:3])

        with tf.GradientTape() as tapet:
            tapet.watch(t)
            with tf.GradientTape() as tapet2:
                tapet2.watch(t)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du0_dt = tapet2.gradient(u[:,0], t)
            du1_dt = tapet.gradient(u[:,1], t)

        with tf.GradientTape() as tapex:
            tapex.watch(x)
            with tf.GradientTape() as tapex2:
                tapex2.watch(x)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du0_dx = tapex2.gradient(u[:,0], x)
            du0_dxx = tapex.gradient(du0_dx, x)

        with tf.GradientTape() as tapey:
            tapey.watch(y)
            with tf.GradientTape() as tapey2:
                tapey2.watch(y)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du0_dy = tapey2.gradient(u[:,0], y)
            du0_dyy = tapey.gradient(du0_dy, y)

        with tf.GradientTape() as tapex:
            tapex.watch(x)
            with tf.GradientTape() as tapex2:
                tapex2.watch(x)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du1_dx = tapex2.gradient(u[:,1], x)
            du1_dxx = tapex.gradient(du1_dx, x)

        with tf.GradientTape() as tapey:
            tapey.watch(y)
            with tf.GradientTape() as tapey2:
                tapey2.watch(y)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du1_dy = tapey2.gradient(u[:,1], y)
            du1_dyy = tapey.gradient(du1_dy, y)

        with tf.GradientTape() as tapey:
            tapey.watch(y)
            with tf.GradientTape() as tapex:
                tapex.watch(x)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du2_dx = tapex.gradient(u[:,2], x)
            du2_dy = tapey.gradient(u[:,2], y)

        f_pred_1 = du0_dt + tf.multiply( u[:,0:1], du0_dx) + tf.multiply(u[:,1:2], du0_dy) + du2_dx - du0_dxx - du0_dyy
        f_pred_2 = du1_dt + tf.multiply( u[:,0:1], du1_dx) + tf.multiply(u[:,1:2], du1_dy) + du2_dy - du1_dxx - du1_dyy

        f_pred = tf.concat((f_pred_1, f_pred_2), axis=1)

        return f_pred

    @tf.function
    def space_model(self, X_f=None):
        """
        Space restrictions required by the NSE (divergence freeness) are implemented here

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
            x = self.x
            y = self.y
        else:
            t = self.tensor(X_f[:, 0:1])
            x = self.tensor(X_f[:, 1:2])
            y = self.tensor(X_f[:, 2:3])

        with tf.GradientTape() as tapey:
            tapey.watch(y)
            with tf.GradientTape() as tapex:
                tapex.watch(x)
                u = self.model(tf.concat([t, x, y], axis = 1))
                du0_dx = tapex.gradient(u[:,0], x)
            du1_dy = tapey.gradient(u[:,1], y)

        div_free = du0_dx + du1_dy
        return div_free

    def get_Lf(self):
        """
        returns growth bound of the Stokes semigroup
        """
        return 0

    def get_M(self):
        """
        returns the scaling factor M for the Stokes semigroup
        """
        return 1.0

    def has_output_not_completely_determined_by_PDE(self):
        return True

    def get_output_dim_completely_determined_by_PDE(self):
        return 2

###############################################################################
## Callouts and Factory Functions
###############################################################################

def load_data(filepath):
   """
   Loads data from csv file specific for the new application
   """
   data = import_csv(filepath, has_headers=False)
   x_data = data[:, 0:3]
   y_data = None
   if data.shape[1]>3:
       y_data = data[:, 3:]
   n_data = data.shape[0]

   return n_data, np.float64(x_data), np.float64(y_data)

def create_pinn(nn_params, lb, ub):
   """
   Factory function to create the NSE PINN with adequate boundary conditions
   """
   bcoffsetlayer = Taylor_BC_OffsetLayer(3,2)
   bcs = BoundaryCondition(i=np.array([1,2]), o=np.array([0,1]), 
                    c=np.array([(ub[1]-lb[1])/2.0, (ub[2]-lb[2])/2.0]), 
                    r=np.array([(ub[1]-lb[1])/2.0, (ub[2]-lb[2])/2.0]), ofs=bcoffsetlayer)
   return NSE(nn_params, lb, ub, bcs=bcs)

def post_train_callout(pinn, output_directory) -> None:
    """
    Callout to be called after training. 

    :param PINN pinn: pinn of type (here) NSE, which has been trained 
    :param string output_directory: path to output directory
    """
    ## number of test points
    n_x = 101
    n_y = 101

    x_lim_low = 0
    x_lim_high = np.pi
    y_lim_low = 0
    y_lim_high = np.pi

    X, Y = np.meshgrid(np.linspace(x_lim_low, x_lim_high, n_x), np.linspace(y_lim_low, y_lim_high, n_y))
    t = np.zeros((n_x*n_y, 1))

    input = np.concatenate([t, np.reshape(X, (n_x*n_y, 1)), np.reshape(Y, (n_x*n_y,1))], axis=1)
    output = pinn.model(input)
    
    ## reference solution
    uref = -np.cos(X)*np.sin(Y)
    vref = np.sin(X)*np.cos(Y)
    pref = -1/4*(np.cos(2*X)+ np.cos(2*Y))

    # export data
    print(input.shape)
    print(output[:,0:2].shape)
    print(np.reshape(uref, (n_x*n_y, 1)).shape)
    print(np.reshape(vref, (n_x*n_y, 1)).shape)
    export_data = np.concatenate([input, output[:,0:2], np.reshape(uref, (n_x*n_y, 1)), np.reshape(vref, (n_x*n_y, 1))], axis = 1)
    export_csv(export_data, os.path.join(output_directory, "train_output.csv"))

    return

def post_extract_callout(input, R, output_directory) -> None:
    """
    Callout to be called after extracting parameters necessary for a posteriori error estimation

    :param np.array input: input data on which the parameter estimation has been taken place
    :param np.array R: approximation error of the ODE/PDE for this input set
    :param np.array delta: smoothened approximation error of the ODE/PDE for this input set
    :param np.array deltadot: derivative of smoothened approximation error of the ODE/PDE for this input set
    :param np.array deltadotdot: second derivative of smoothened approximation error of the ODE/PDE for this input set
    :param string output_directory: path to output directory
    """
    
    return

def post_eval_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None , tail=None) -> None:

    """
    Callout to be called after running NN for a testdataset X_data

    :param string outdir: path to output directory
    :param np.array X_data: input data set on which the NN has been evaluated
    :param np.array Y_pred: prediction of NN of system state
    :param np.array E_pred: predicted error (two components if numerical integration is used)
    :param np.array N_SP_pred: predicted number of supportpoints for numerical integration
    :param np.array Y_data: reference output data set (optional)

    """
        
    error =  np.sum((Y_pred[:,0:2] - Y_data[:, 0:2])**2, axis = 1)
    error = np.reshape(error, (error.shape[0],1))
    I1, _, E = compute_l2_norm(error, X_data[:,1:3], outdim=2)

    data = {}
    data['E_init'] = E_pred[0]
    data['E_PI'] = E_pred[1]
    data['N_SP'] = int(N_SP_pred)
    data['E_ref'] = np.sqrt(I1+ E)

    print(os.path.join(outdir, "run_" + get_prefix() +tail[:-4] + "_error.json"))
    out_file = open( os.path.join(outdir,"..", "run_"+get_prefix()+"values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()
    
    return
