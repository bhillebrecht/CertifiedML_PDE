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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from base.pinn import PINN
from helpers.csv_helpers import import_csv
from helpers.globals import get_prefix
from helpers.input_generator import generate_collocation_points
from helpers.integrate import compute_integral_trpz
from helpers.nn_parametrization import get_param_as_float, has_param
from helpers.plotting import new_fig, save_fig

###############################################################################
## Klein Gordon PINN
###############################################################################
class KleinGordon(PINN):

    def __init__(self, layers, lb, ub, X_f=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub)
        self.x = None
        self.t = None

        self.x_Neumann = None
        self.t_Neumann = None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
        self.t = self.tensor(X_f[:,0:1])
        self.x = self.tensor(X_f[:,1:2])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the considered ODE/PDE

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            t = self.t
            x = self.x
        else:
            t = self.tensor(X_f[:,0:1])
            x = self.tensor(X_f[:,1:2])

        with tf.GradientTape() as tapet:
            tapet.watch(t)
            with tf.GradientTape() as tapet2:
                tapet2.watch(t)
                u = self.model(tf.concat([t,x], axis=1))
                u1_t = tapet2.gradient(u[:,1], t)
            u0_t = tapet.gradient(u[:,0], t)

        with tf.GradientTape() as tapex2:
            tapex2.watch(x)
            with tf.GradientTape() as tapex1:
                tapex1.watch(x)
                u = self.model(tf.concat([t, x], axis = 1))
                u1_x = tapex1.gradient(u[:,1], x)
            u1_xx = tapex2.gradient(u1_x, x)

        # Klein Gordan as system of first order PDEs
        # u2: true value
        # u1: velocity
        # 0: v_t = u_xx - 1/4 u
        # 1: u_t = v

        f_pred_0 = u0_t - u1_xx + 0.25*u[:,1:2]
        f_pred_1 = u1_t - u[:,0:1]

        return tf.concat([f_pred_0, f_pred_1], axis = 1)

    def set_neumann_boundary(self, x):
        self.t_space = self.tensor(x[:,0:1])
        self.x_space = self.tensor(x[:,1:2])
        self.has_space_restriction = True
        self.space_weight = 50
        return

    def space_model(self):

        tint = self.t_space
        xint = self.x_space

        with tf.GradientTape() as tapex:
            tapex.watch(xint)
            u = self.model(tf.concat([tint, xint], axis=1))
            u1x = tapex.gradient(u[:,1], xint)

        with tf.GradientTape() as tapex:
            tapex.watch(xint)
            u = self.model(tf.concat([tint, xint], axis=1))
            u0x = tapex.gradient(u[:,0], xint)

        f_space = tf.concat([u0x, u1x], axis=0)

        return f_space

    def get_Lf(self):
        """
        returns lipschitz constant or spectral abscissa of right hand side of ODE
        """
        return 0

    def get_M(self):
        return 164.43

    def has_hard_constraints(self):
        return True

###############################################################################
## Callouts and Factory Functions
###############################################################################

def load_data(filepath):
   """
   Loads data from csv file specific for the new application
   """
   data = import_csv(filepath, has_headers=True)
   x_data = data[:, 0:2]
   y_data = None
   if data.shape[1]>2:
       y_data = data[:, 2:4]
   n_data = data.shape[0]

   return n_data, x_data, y_data

def create_pinn(nn_params, lb, ub):
   """
   Factory function to create the target PINN 
   """
   pinn = KleinGordon(nn_params, lb, ub)
   times = generate_collocation_points(500, lb[0:1], ub[0:1])
   neumann = np.concatenate( [np.concatenate([times, np.ones(times.shape)*lb[1]], axis=1), np.concatenate([times, np.ones(times.shape)*ub[1]], axis=1)], axis=0)
   pinn.set_neumann_boundary(neumann)
   return pinn

def post_train_callout(pinn, output_directory) -> None:
    """
    Callout to be called after training. 

    :param PINN pinn: pinn of type (here) TemplateAppl, which has been trained 
    :param string output_directory: path to output directory
    """

    t = np.zeros((100,1))
    x = np.reshape(np.linspace(0,1,100), (100,1))

    f_0 = np.cos(2*np.pi*x)
    f_1 = 0.5*np.cos(4*np.pi*x)#+ np.cos(8*np.pi*x)

    input = np.concatenate((t, x), axis=1)

    y_pred, _ = pinn.predict(input)

    cmap = plt.get_cmap("RdBu")
    blue = cmap(1.0)
    lightblue = cmap(.8)    
    red = cmap(0.0)
    lightred = cmap(.2)

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    ax.set( ylabel=r'$u(t,x)$')
    ax.set( xlabel="$x$")

    ax.grid(True, which='both')
    ax.set(xlim=[0,1])

    ax.plot(x, y_pred[:,0], linewidth=1.5, color=blue, linestyle="-", label="$u_{pinn}$")
    ax.plot(x, y_pred[:,1], linewidth=1.5, color=red, linestyle="-", label="$\\partial_t u_{pinn}$")

    ax.plot(x, f_1, linewidth=1.5, color=lightblue, linestyle=":", label="$u_{ref}$")
    ax.plot(x, f_0, linewidth=1.5, color=lightred, linestyle=":", label="$\\partial_t u_{ref}$")

    ax.legend(loc='best')
    fig.tight_layout()

    #plt.show()
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

    error =  (Y_pred[:,0] - Y_data[:,0])**2 + (Y_pred[:,1] - Y_data[:,1])**2
    I1 = compute_integral_trpz(error, X_data[1,1]-X_data[0,1])
    I2 = compute_integral_trpz(error[::2], X_data[2,1]-X_data[0,1])

    #determine second error in H^2
    e1 = (Y_pred[:,0] - Y_data[:,0])
    Ienwnorm = compute_integral_trpz(0.25*(Y_pred[:,0] - Y_data[:,0])**2 , X_data[1,1]-X_data[0,1])\
             + compute_integral_trpz(0.25*(Y_pred[:,0] - Y_data[:,0])**2 , X_data[1,1]-X_data[0,1])\
             + compute_integral_trpz((1/(X_data[1,1]- X_data[0,1])*np.diff(e1,2))**2, X_data[1,1]-X_data[0,1])

    print(Ienwnorm)

    data = {}
    data['E_init'] = E_pred[0]
    data['E_PI'] = E_pred[1]
    data['N_SP'] = int(N_SP_pred)
    data['E_ref'] = np.sqrt(I1)
    data['E_ref_newnorm'] = np.sqrt(Ienwnorm)
    if has_param(os.path.join(outdir, "..", "run_tanh_values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "E_bc"):
        data['E_bc'] = get_param_as_float(os.path.join(outdir, "..", "run_tanh_values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "E_bc")

    out_file = open( os.path.join(outdir, "..", "run_tanh_values", "run_" + get_prefix() +tail[:-4] + "_error.json"), "w")
    json.dump(data, out_file, indent=4)
    out_file.close()
    return

