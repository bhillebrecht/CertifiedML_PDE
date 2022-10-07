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
import pandas


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from base.pinn import PINN
from base.periodicity_layers import PeriodicLayers
from helpers.globals import get_prefix
from helpers.csv_helpers import import_csv
from helpers.l2_norm import compute_l2_norm
from helpers.plotting import new_fig, save_fig

###############################################################################
## 2-dimensional transport equation modelled as PINN
###############################################################################
class TransportEquation(PINN):

    def __init__(self, layers, lb, ub, p, X_f=None ):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub, p, normalize=False)
        self.t = None
        self.x = None
        self.y = None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
      self.t = self.tensor(X_f[:,0:1])
      self.x = self.tensor(X_f[:,1:2])
      self.y = self.tensor(X_f[:,2:3])

    @tf.function
    def f_model(self, X_f=None):
        """
        Function defining the physics informed loss for the transport equation

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
            with tf.GradientTape() as tapex:
                tapex.watch(x)
                with tf.GradientTape() as tapey:
                    tapey.watch(y)
                    u = self.model(tf.concat((t, x, y), axis=1))
                    dq_dt = tapet.gradient(u, t)
                    dq_dy = tapey.gradient(u, y)
                    dq_dx = tapex.gradient(u, x)
        f_pred = dq_dt- 0.2*dq_dx - 0.5*dq_dy
        return f_pred

    def get_Lf(self):
        """
        returns lipschitz constant or spectral abscissa of right hand side of ODE
        """
        return 0

    def get_M(self):
        return 1.0


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
       y_data = data[:, 3:4]
   n_data = data.shape[0]

   return n_data, np.float64(x_data), np.float64(y_data)

def create_pinn(nn_params, lb, ub):
   """
   Factory function to create the target PINN 
   """
   periodicity = PeriodicLayers ( np.array([[0,40], [4, 40], [4,40]], dtype="float64"), 20)
   return TransportEquation(nn_params, lb, ub, periodicity)

def post_train_callout(pinn, output_directory) -> None:
    """
    Callout to be called after training. 

    :param PINN pinn: pinn of type (here) TemplateAppl, which has been trained 
    :param string output_directory: path to output directory
    """
    n_x = 201
    n_y = 201
    
    t0 = np.zeros((n_x*n_y,1))
    t1 = np.ones((n_x*n_y, 1))
    
    x = np.reshape(np.linspace(-2,2,n_x), (n_x,1)) 
    x = np.repeat(x, n_y)
    x = np.reshape(x, (n_x*n_y, 1))
    
    y = np.reshape(np.linspace(-2,2,n_y), (n_y,1)) 
    y = np.repeat(y, n_x)
    y = np.reshape(y, (n_y, n_x))
    y = y.transpose()
    y = np.reshape(y, (n_x*n_y, 1))

    input0 = np.concatenate((t0, x, y), axis=1)
    input1 = np.concatenate((t1, x, y), axis=1)
    input2 = np.concatenate((t1*2, x, y), axis=1)
    input4 = np.concatenate((t1*4, x, y), axis=1)

    z_pred, _ = pinn.predict(input0)
    z_pred1, _ = pinn.predict(input1)
    z_pred2, _ = pinn.predict(input2)
    z_pred4, _ = pinn.predict(input4)

    fig = new_fig()

    ax = fig.add_subplot(221)
    ax.set( ylabel="$y$")
    ax.grid(False, which='both')
    ax.set(xlim=[-2,2])
    ax.set(ylim=[-2,2])
    c = ax.pcolormesh(np.reshape(x, (n_x, n_y)), np.reshape(y, (n_x, n_y)), np.reshape(z_pred, (n_x, n_y)), cmap='Blues', vmin=0, vmax=0.5, label="$t=0$")
    fig.colorbar(c, ax=ax)
    ax.text(-0.15, 1.08, "a) $t=0$", transform=ax.transAxes, size=10)
            
    ax = fig.add_subplot(222)
    ax.grid(False, which='both')
    ax.set(xlim=[-2,2])
    ax.set(ylim=[-2,2])
    fig.colorbar(c, ax=ax)
    c = ax.pcolormesh(np.reshape(x, (n_x, n_y)), np.reshape(y, (n_x, n_y)), np.reshape(z_pred1, (n_x, n_y)), cmap='Blues', vmin=0, vmax=0.5, label="$t=1$")
    ax.text(-0.15, 1.08, "b) $t=1$", transform=ax.transAxes, size=10)

    ax = fig.add_subplot(223)
    ax.set( ylabel="$y$")
    ax.set( xlabel="$x$")
    ax.grid(False, which='both')
    ax.set(xlim=[-2,2])
    ax.set(ylim=[-2,2])
    fig.colorbar(c, ax=ax)
    c = ax.pcolormesh(np.reshape(x, (n_x, n_y)), np.reshape(y, (n_x, n_y)), np.reshape(z_pred2, (n_x, n_y)), cmap='Blues', vmin=0, vmax=0.5, label="$t=2$")
    ax.text(-0.15, 1.08, "c) $t=2$", transform=ax.transAxes, size=10)

    ax = fig.add_subplot(224)
    ax.set( xlabel="$x$")
    ax.grid(False, which='both')
    ax.set(xlim=[-2,2])
    ax.set(ylim=[-2,2])
    c = ax.pcolormesh(np.reshape(x, (n_x, n_y)), np.reshape(y, (n_x, n_y)), np.reshape(z_pred4, (n_x, n_y)), cmap='Blues', vmin=0, vmax=0.5, label="$t=4$")
    ax.text(-0.15, 1.08, "d) $t=4$", transform=ax.transAxes, size=10)
    fig.colorbar(c, ax=ax)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    plt.show()
    save_fig(fig, "output_training" , output_directory)

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

def post_eval_callout(outdir, X_data, Y_pred, E_pred, N_SP_pred, Y_data=None, tail="") -> None:

    """
    Callout to be called after running NN for a testdataset X_data

    :param string outdir: path to output directory
    :param np.array X_data: input data set on which the NN has been evaluated
    :param np.array Y_pred: prediction of NN of system state
    :param np.array E_pred: predicted error (two components if numerical integration is used)
    :param np.array N_SP_pred: predicted number of supportpoints for numerical integration
    :param np.array Y_data: reference output data set (optional)

    """

    t = X_data[0,0]
    fig = new_fig(ratio=0.75)
    ax = fig.add_subplot(1,1,1)
    ax.set( xlim=[-2,2])
    ax.set( ylim=[-2,2])
    ax.set_title("$\hat{u}( t,x, y)$; $t = "+ "{:3.2f}".format(t)+ "$")

    data = pandas.DataFrame(data={'x':X_data[:,1], 'y':X_data[:,2], 'z': np.reshape(Y_pred[:,:] , X_data[:,2].shape)})
    data = data.pivot(index='y', columns='x', values='z')

    I1, _, E = compute_l2_norm((Y_pred[:,0] - Y_data[:,0]), X_data[:,1:3])

    data = {}
    data['E_init'] = E_pred[0]
    data['E_PI'] = E_pred[1]
    data['N_SP'] = int(N_SP_pred)
    data['E_ref'] = I1

    out_file = open( os.path.join(outdir, "..", "run_tanh_values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()

    return

