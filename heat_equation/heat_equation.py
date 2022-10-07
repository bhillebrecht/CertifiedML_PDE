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

from helpers.csv_helpers import import_csv
from helpers.globals import get_prefix
from helpers.l2_norm import compute_l2_norm
from helpers.plotting import new_fig, save_fig

###############################################################################
## PINN modelling the solution to the 1-dimensional heat equation
###############################################################################
class HeatEquation(PINN):

    def __init__(self, layers, lb, ub, X_f=None, bcs=None):
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

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
      self.t = self.tensor(X_f[:,0:1])
      self.x = self.tensor(X_f[:,1:2])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the heat equation

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
            x = self.x
        else:
            t = self.tensor(X_f[:, 0:1])
            x = self.tensor(X_f[:, 1:2])

        with tf.GradientTape() as tapet:
            tapet.watch(t)
            with tf.GradientTape() as tapex:
                tapex.watch(x)
                with tf.GradientTape(persistent=True) as tapex2:
                    tapex2.watch(x)
                    q = self.model(tf.concat((t, x), axis=1), training=False)
                dq_dx = tapex2.gradient(q, x)
            dq_dxx = tapex.gradient(dq_dx, x)
        dq_dt = tapet.gradient(q, t)
        del tapex2
        f_pred = dq_dt-0.2*dq_dxx

        del tapex
        return f_pred

    def get_Lf(self):
        """
        returns the growth bound of the heat semigroup
        """
        return 0 # contractive
        return  (- 0.2*np.pi**2) # expdecay

    def get_M(self):
        """
        returns the scaling parameter of the heat semigroup
        """
        return 1.0

###############################################################################
## Callouts and Factory Functions
###############################################################################

def load_data(filepath):
   """
   Loads data from csv file specific for the new application
   """
   data = import_csv(filepath)
   x_data = data[:, 0:2]
   y_data = None
   if data.shape[1]>2:
       y_data = data[:, 2:3]
   n_data = data.shape[0]

   return n_data, np.float64(x_data), np.float64(y_data)

def create_pinn(nn_params, lb, ub):
   """
   Factory function to create the target PINN 
   """
   bcs = BoundaryCondition(np.array([1]), np.array([0]), np.array([np.float64(0.5)]), np.array([np.float64(0.5)]))
   return HeatEquation(nn_params, lb, ub, bcs=bcs)

def post_train_callout(pinn, output_directory) -> None:
    """
    Callout to be called after training. 

    :param PINN pinn: pinn of type (here) TemplateAppl, which has been trained 
    :param string output_directory: path to output directory
    """
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

def post_eval_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None, tail=None ) -> None:

    """
    Callout to be called after running NN for a testdataset X_data

    :param string outdir: path to output directory
    :param np.array X_data: input data set on which the NN has been evaluated
    :param np.array Y_pred: prediction of NN of system state
    :param np.array E_pred: predicted error (two components if numerical integration is used)
    :param np.array N_SP_pred: predicted number of supportpoints for numerical integration
    :param np.array Y_data: reference output data set (optional)

    """
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.set( ylabel=r'$u(t,x)$')
    ax.set( xlabel="$x$")

    ax.grid(True, which='both')
    ax.set(xlim=[0,1])
    ax.set(ylim=[-1.1,1.1])

    ax.plot(X_data[:,1:2], Y_pred[:,:], linewidth=1.5, color=colors[0], linestyle="-", label="pred")
    ax.plot(X_data[:,1:2], Y_data[:,:], linewidth=1.5,  color=colors[2], linestyle="-.", label="ref")

    ax.legend(loc='best')
    fig.tight_layout()

    save_fig(fig, "output_"+ str(X_data[0,0]) , outdir)
    
    data = {}
    data['E_ref'], _, _ = compute_l2_norm((Y_pred[:,0] - Y_data[:,0]), X_data[:,1:2])
    data['E_init'] = E_pred[0]
    data['E_PI'] = E_pred[1]
    data['N_SP'] = int(N_SP_pred)

    out_file = open( os.path.join(outdir, "..", "run_"+get_prefix()+"values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()

    return

