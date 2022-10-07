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

import tensorflow as tf

from base.pinn import PINN
from helpers.csv_helpers import import_csv

###############################################################################
## Template Application for PINN
##    it is mandatory to fill the following functions:
##    - __init__
##    - set_collocation_points
##    - f_model
##    - get_Lf
###############################################################################
class TemplateAppl(PINN):

    def __init__(self, layers, lb, ub, X_f=None):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bound of the inputs of the training data
        :param np.ndarray ub: upper bound of the inputs of the training data
        :param np.ndarray X_f: collocation points
        """

        super().__init__(layers, lb, ub)
        self.x0 = None

        if X_f is not None:
            self.set_collocation_points(X_f)

    def set_collocation_points(self, X_f):
      self.x0 = self.tensor(X_f[:,0:1])

    @tf.function
    def f_model(self, X_f=None):
        """
        The actual PINN to approximate the motion of the considered ODE/PDE

        :return: tf.Tensor: the prediction of the PINN
        """

        if X_f is None:
            x0 = self.x0
        else:
            x0 = self.tensor(X_f[:,0:1])

        f_pred = tf.zeros(x0.shape)
        return f_pred

    def get_Lf(self):
        """
        returns lipschitz constant or spectral abscissa of right hand side of ODE
        """
        return 150

    def get_M(self):
        return 1.0

###############################################################################
## Callouts and Factory Functions
###############################################################################

def load_data(filepath):
   """
   Loads data from csv file specific for the new application
   """
   data = import_csv(filepath)
   x_data = data[:, 0:1]
   y_data = None
   if data.shape[1]>1:
       y_data = data[:, 1:2]
   n_data = data.shape[0]

   return n_data, x_data, y_data

def create_pinn(nn_params, lb, ub):
   """
   Factory function to create the target PINN 
   """
   return TemplateAppl(nn_params, lb, ub)

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
    return

