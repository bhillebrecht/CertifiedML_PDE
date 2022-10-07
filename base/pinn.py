###################################################################################################
#   Copyright (c) 2022 Birgit Hillebrecht
#
#   To cite this code in publications, please use either
#    - B. Hillebrecht and B. Unger : "Certified machine learning: Rigorous a posteriori error bounds for PDE defined PINNs", arxiV preprint available
#    - B. Hillebrecht and B. Unger, "Certified machine learning: A posteriori error 
#      estimation for physics-informed neural networks," 2022 International Joint
#      Conference on Neural Networks (IJCNN), 2022, pp. 1-8, 
#      doi: 10.1109/IJCNN55064.2022.9892569.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
###################################################################################################

import abc
import tensorflow as tf

from base.nn import NN
from helpers.globals import get_w_adaptivity, get_w_adaptivity_factor, get_w_data


class PINN(NN, metaclass=abc.ABCMeta):
    """
    Class used to represent a Physics informed Neural Network, children of NN.
    """

    def __init__(self, layers, lb, ub, periodicity = None, bcs = None, normalize=True):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        super().__init__(layers, lb, ub, periodicity, bcs, normalize)

        self.loss_object = self.loss
        self.use_adaptive_weights = get_w_adaptivity()

        self.w_data = get_w_data() / (1 + get_w_data() )
        self.w_phys = 1 / (1 + get_w_data() )

        self.alpha = get_w_adaptivity_factor()

        self.adpative_constant_bcs_list = []
        self.dalpha = 20

        self.has_space_restriction = False
        self.space_weight = 0
        
        if self.use_adaptive_weights:
            with open('weights.csv','a') as fd:
                fd.write("New run\n")

    @tf.function
    def train_step(self, x, y):
        """
        Performs training step during training.

        :param tf.tensor x: (batched) input tensor of training data
        :param tf.tensor y: (batched) output tensor of training data
        :return: float loss: the corresponding current loss value
        """
        if self.use_adaptive_weights:
            with tf.GradientTape() as tape:
                if self.dalpha > 0.00001:
                    loss = self.theotherloss(x, y)
                else:
                    y_pred = self.model(x)
                    loss = self.loss_object(y, y_pred)
        else:
            with tf.GradientTape() as tape:
                y_pred = self.model(x)
                loss = self.loss_object(y, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def theotherloss(self, x, y):
        # Collect derivatives of the loss w.r.t. 
        self.grad_res = []
        self.grad_bcs = []
        for i in range(len(self.layers) - 1):
            currentweight = self.model.weights[i]
            with tf.GradientTape() as tape1:
                tape1.watch(currentweight)
                y_pred = self.model(x)
                L_data = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), axis=1)) 
                self.grad_bcs.append(tape1.gradient(L_data, currentweight))
           
            with tf.GradientTape() as tape2:
                tape2.watch(currentweight)
                f_pred = self.f_model()
                L_phys = tf.reduce_mean(tf.reduce_sum(tf.square(f_pred), axis=1)) 
                self.grad_res.append(tape2.gradient(L_phys, currentweight))

        for i in range(len(self.layers) - 1):
            if  tf.reduce_mean(tf.abs(self.grad_bcs[i])) != 0:
                self.adpative_constant_bcs_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs[i])))

        # compute new data weight
        lambda_hat = tf.reduce_max(tf.stack(self.adpative_constant_bcs_list))    
        self.w_data = self.w_data*(1.0-self.alpha) + self.alpha*lambda_hat
        self.dalpha = tf.abs(lambda_hat- self.w_data) 

        # norm total weights to 1 again
        self.w_data = self.w_data/(1.0 + self.w_data)
        with open('weights.csv','a') as fd:
            fd.write(str(self.w_data.numpy())+ "\n")

        # determine loss
        L = self.w_data*L_data  + self.w_phys*L_phys
        return L

    def loss(self, y, y_pred):
        """
        Customized loss object to represent the composed mean squared error
        for Physics informed Neural Networks.
        Consists of the mean squared error 
            between the predictions from the DNN (model) and reference values of the solution from the differential equation
            between and the mean squared error of the predictions of the PINN (f_model) with zero.

        :param tf.tensor y: reference values of the solution of the differential equation
        :param tf.tensor y_pred: predictions of the solution of the differential equation
        :return: tf.tensor: composed mean squared error value
        """

        f_pred = self.f_model()

        dataerr = tf.square(y - y_pred)

        L_data = tf.reduce_mean(tf.reduce_sum(dataerr, axis=1)) 
        L_phys = tf.reduce_mean(tf.reduce_sum(tf.square(f_pred), axis=1)) 

        L = self.w_data*L_data  + self.w_phys*L_phys

        if self.has_space_restriction:
            s_pred = self.space_model()
            L_space = tf.reduce_mean(tf.reduce_sum(tf.square(s_pred), axis=1))
            L = L + self.space_weight * L_space

            
        return L

    def f_model(self, x):
        """
        Declaration of the function for the implementation of the f_model for a specific differential equation.
        """
        pass


    def space_model(self):
        """
        Declaration of the function for the implementation of the space_model for a specific differential equation.
        """
        pass

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        if x.shape[1] == self.input_dim :
            return self.model.predict(x), self.f_model(x)
        else :
            return self.model.predict(x[:,0:self.input_dim]), self.f_model(x[:,0:self.input_dim])

    def has_hard_constraints(self):
        return True

    def get_ISS_param(self):
        return 0

    def has_output_not_completely_determined_by_PDE(self):
        return False
    
    def get_output_dim_completely_determined_by_PDE(self):
        return 0