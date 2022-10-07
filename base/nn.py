###################################################################################################
#
# This file incorporates work and modifications to the originally published code
# according to the previous license by the following contributors under the following licenses
#
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
# Copyright (c) 2021 Jonas Nicodemus
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

import abc
import logging
import os
import time
import datetime

from pathlib import Path
from base.bc_layer import AddOffsetLayer, BoundaryCondition, LengthFactor, MultiplyLengthFactor
from helpers.globals import get_activation_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from base.lbfgs import LBFGS

from base.periodicity_layers import NonPeriodic_Base, Periodic_Base, PeriodicLayers

CHECKPOINTS_PATH = os.path.join('../checkpoints')

class NN(object, metaclass=abc.ABCMeta):
    """
    Abstract class used to represent a Neural Network.
    """

    def __init__(self, layers: list, 
        lb: np.ndarray,
        ub: np.ndarray,
        periodicity:PeriodicLayers = None,
        bcs:BoundaryCondition = None,
        normalize = True) -> None:
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """
        tf.config.run_functions_eagerly(True) 
        self.checkpoints_dir = CHECKPOINTS_PATH
        self.dtype = "float64"

        self.input_dim = layers[0]
        self.output_dim = layers[-1]
        self.lb = lb
        self.ub = ub
        self.layers = layers

        self.model = self.get_model(periodicity, lb, ub, bcs, normalize)
        # print(self.model.summary())

        self.optimizer = None
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.start_time = None
        self.prev_time = None

        # Store metrics
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_time_results = {}
        self.train_pred_results = {}

        self.w_data = 0

    def get_model(self, periodicity, lb, ub, bcs: BoundaryCondition, normalize=True):
        res_net = False
        skip_param = 3

        # use the functional API 
        inputlayer = layers.Input(self.input_dim, dtype='float64')

        if bcs is not None and periodicity is not None:
            logging.warning("Periodic bcs and enforced bcs were given. Ensure that this is a valid model.")

        if bcs is not None:
            # logging.info
            logging.info("Periodic BC enforcement.")
            xbc = LengthFactor(input_dim=self.input_dim, units=bcs.OUTPUT_INDICES.shape[0], 
                ccenter=bcs.CENTER, radius=bcs.RADIUS, ind=bcs.INPUT_INDICES) (inputlayer)

            if bcs.OffsetLayer is not None:
                xbc_offset = (bcs.OffsetLayer)(inputlayer)
        else:
            logging.info("No detour for hard boundary constraints added")

        # initialize working content
        x = inputlayer 

        # Periodicity Layer
        if periodicity is not None:
            # create periodicity model list
            PML = []

            # create linear model list
            LML = []

            # count periodic layers
            t_output_dim = 0
            tdc_old = 0
            tdc = []
            for index in range(self.input_dim):
                if periodicity.index_and_n[index,0] != 0:
                    tdc_old = t_output_dim
                    t_output_dim = t_output_dim + periodicity.index_and_n[index,1]*3
                    tdc.append([tdc_old+self.input_dim, t_output_dim+self.input_dim])
            tdc = np.array(tdc)

            # detour to compute phi, a, c for periodic base
            t_detour = layers.Lambda(lambda X: tf.reshape(X[:, 0], (tf.shape(X)[0], 1)), dtype="float64") (x)
            t_detour = layers.Dense(t_output_dim, activation=get_activation_function(), kernel_initializer="glorot_uniform", dtype="float64")(t_detour)
            t_detour = layers.Dense(t_output_dim, activation=get_activation_function(), kernel_initializer="glorot_uniform", dtype="float64")(t_detour)
            x = layers.concatenate([x, t_detour], axis=1, dtype="float64")

            # fill layers
            t_detour_counter = 0
            for index in range(self.input_dim):
                if periodicity.index_and_n[index,0] != 0:
                    logging.info("Periodic at index " + str(index) + " with " + str(periodicity.index_and_n[index,1]) +" units.")
                    pbase = Periodic_Base(input_dim=self.input_dim, units=periodicity.index_and_n[index,1], period=periodicity.index_and_n[index,0], ind=index, tdc=tdc[t_detour_counter])(x)
                    PML.append(pbase)
                else:
                    logging.info("Linear at index " + str(index) + " with " + str(periodicity.index_and_n[index,1]) +" units.")
                    slice = NonPeriodic_Base(input_dim=self.input_dim, units=periodicity.index_and_n[index,1], ind=index)(x)
                    LML.append(slice)
    
            # concatenate layers
            concatenated_PML = None
            for item in PML:
                if concatenated_PML is None:
                    concatenated_PML = item
                else:
                    concatenated_PML = layers.concatenate([concatenated_PML, item])

            concatenated_LML = None
            for item in LML:
                if concatenated_LML is None:
                    concatenated_LML = item
                else:
                    concatenated_PML = layers.concatenate([concatenated_LML, item])
        
            x = layers.concatenate([concatenated_LML, concatenated_PML], axis=1, dtype="float64")
            x = layers.Dense(periodicity.m, activation=get_activation_function(),
                            bias_initializer='glorot_normal', kernel_initializer='glorot_normal', dtype="float64")(x)
        else:
            logging.info("No periodic layers added")

        # Normalize input
        if normalize:
            x = layers.Lambda(lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0)(x)

        # neural network itself:
        layer_count = 0
        for layer_width in self.layers[1:-1]:
            if res_net:
                if layer_count == 1:
                    y = x
                if (layer_count+1) % skip_param == 0 and layer_count > 0:
                    x = layers.Lambda(lambda X: X, dtype="float64")(y) + layers.Dense(units = layer_width, activation=get_activation_function(), dtype="float64")(x)
                    y = x
                else:
                    x = layers.Dense(units = layer_width, activation=get_activation_function(), dtype="float64")(x)
                layer_count = layer_count + 1
            else: 
                x = layers.Dense(units = layer_width, activation=get_activation_function(), dtype="float64")(x)
        # Output Layer :
        output = layers.Dense(self.output_dim, dtype="float64")(x)

        if bcs is not None:
            logging.info("Add multiply layer...")
            output = layers.concatenate([output, xbc], axis=1, dtype="float64")
            output = MultiplyLengthFactor(input_dim=self.output_dim + bcs.OUTPUT_INDICES.shape[0], units=self.output_dim, oi = bcs.OUTPUT_INDICES)(output)

            if bcs.OffsetLayer is not None:
                output = layers.concatenate([output, xbc_offset], axis = 1, dtype="float64")
                output = AddOffsetLayer(input_dim=self.output_dim + bcs.OUTPUT_INDICES.shape[0], units=self.output_dim, oi = bcs.OUTPUT_INDICES)(output)
        else:
            logging.info("No multiplication layer for exact Dirichlet BCs added")

        model = tf.keras.Model(inputs=inputlayer, outputs=output)
        return model

    def tensor(self, X):
        """
        Converts a list or numpy array to a tf.tensor.

        :param list or nd.array X:
        :return: tf.tensor: tensor of X
        """
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def summary(self):
        """
        Pipes the Keras.model.summary function to the logging.
        """

        self.model.summary(print_fn=lambda x: logging.info(x))

    @tf.function
    def train_step(self, x, y):
        """
        Performs training step during training.

        :param tf.tensor x: (batched) input tensor of training data
        :param tf.tensor y: (batched) output tensor of training data
        :return: float loss: the corresponding current loss value
        """

        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.loss_object(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def fit(self, x, y, epochs=2000, x_test=None, y_test=None, optimizer='adam', learning_rate=0.1,
            load_best_weights=False, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the neural network training phase.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate current accuracy
        :param str optimizer: name of the optimizer, choose from 'adam' or 'lbfgs'
        :param bool load_best_weights: flag to determine if the best weights corresponding to the best
        accuracy are loaded after training
        """

        x = self.tensor(x)
        y = self.tensor(y)
        

        self.start_time = time.time()
        self.prev_time = self.start_time

        if optimizer == 'adam':
            self.train_adam(x, y, epochs, x_test, y_test, learning_rate, val_freq, log_freq, verbose)
        elif optimizer == 'lbfgs':
            self.train_lbfgs(x, y, epochs, x_test, y_test, learning_rate, val_freq, log_freq, verbose)

        if load_best_weights is True:
            self.load_weights()

    def train_adam(self, x, y, epochs=2000, x_test=None, y_test=None, learning_rate=0.1, val_freq=1000, log_freq=1000,
                   verbose=1):
        """
        Performs the neural network training, using the adam optimizer.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        epoch_loss = tf.keras.metrics.Mean(name='epoch_loss')
        if verbose:
            logging.info(f'Start ADAM optimization')

        for epoch in range(1, epochs + 1):
            loss = self.train_step(x, y)

            # Track progress
            epoch_loss.update_state(loss)  # Add current batch loss

            self.epoch_callback(epoch, epoch_loss.result(), epochs, x_test, y_test, val_freq, log_freq,
                                verbose)

    def train_lbfgs(self, x, y, epochs=2000, x_test=None, y_test=None, learning_rate=1.0, val_freq=1000, log_freq=1000,
                    verbose=1):
        """
        Performs the neural network training, using the L-BFGS optimizer.

        :param tf.tensor x: input tensor of the training dataset
        :param tf.tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param tf.tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param tf.tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        # train the model with L-BFGS solver
        if verbose:
            logging.info(f'Start L-BFGS optimization')

        optimizer = LBFGS()
        optimizer.minimize(
            self.model, self.loss_object, x, y, self.epoch_callback, epochs, x_test=x_test, y_test=y_test,
            val_freq=val_freq, log_freq=log_freq, verbose=verbose, learning_rate=learning_rate)

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model.predict(x)

    def train_results(self):
        """
        Returns the training metrics stored in dictionaries.

        :return: dict: loss over epochs, dict: accuracy over epochs,
        dict: predictions (on the testing dataset) over epochs
        """

        return self.train_loss_results, self.train_accuracy_results, self.train_pred_results

    def reset_train_results(self):
        """
        Clears the training metrics.
        """
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_pred_results = {}

    def get_weights(self):
        """
        Returns the model weights.

        :return: tf.tensor model weights
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def save_weights(self, path):
        """
        Saves the model weights under a specified path.

        :param str path: path where the weights are saved
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path)

    def load_weights(self, path=None):
        """
        Loads the model weights from a specified path.

        :param str path: path where the weights are saved,
        if None the weights are assumed to be saved at the checkpoints directory
        """

        if path is None:
            path = self.checkpoints_dir

        self.model.load_weights(tf.train.latest_checkpoint(path))
        logging.info(f'\tWeights loaded from {path}')
        
    def get_epoch_duration(self):
        """
        Measures the time for a training epoch.

        :return: float time per epoch
        """

        now = time.time()
        epoch_duration = datetime.datetime.fromtimestamp(now - self.prev_time).strftime("%M:%S.%f")[:-4]
        self.prev_time = now
        return epoch_duration

    def get_elapsed_time(self):
        """
        Measures the time since training start.

        :return: float elapsed time
        """

        return datetime.timedelta(seconds=int(time.time() - self.start_time))

    def epoch_callback(self, epoch, epoch_loss, epochs, x_val=None, y_val=None, val_freq=1000, log_freq=1000,
                       verbose=1):
        """
        Callback function, which is called after each epoch, to produce proper training logging
        and keep track of training metrics.

        :param int epoch: current epoch
        :param float epoch_loss: current loss value
        :param int epochs: number of training epochs
        :param tf.tensor x_val: input tensor of the test dataset, used to evaluate current accuracy
        :param tf.tensor y_val: output tensor of the test dataset, used to evaluate current accuracy
        :param int val_freq: number of epochs passed before trigger validation
        :param int log_freq: number of epochs passed before each logging
        """
        self.train_loss_results[epoch] = epoch_loss
        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))

            if epoch > val_freq:
                rel_improv = 100*(1-epoch_loss/self.train_loss_results[epoch-val_freq])
            else:
                rel_improv = -1
                
            log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                      f'Loss: {epoch_loss:.4e}, \t Rel.Improv [%]: {rel_improv:.2f}'

            if x_val is not None and y_val is not None:
                [mean_squared_error, errors, Y_pred] = self.evaluate(x_val, y_val)
                self.train_accuracy_results[epoch] = mean_squared_error
                self.train_pred_results[epoch] = Y_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error:.4e}'
                if mean_squared_error <= min(self.train_accuracy_results.values()):
                    self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))

            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += f',\t Elapsed time: {elapsed_time} (+{self.get_epoch_duration()})'
                logging.info(log_str)

        if epoch == epochs and x_val is None and y_val is None:
            self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))

    def evaluate(self, x_val, y_val, metric='MSE'):
        """
        Calculates the accuracy on a testing dataset.

        :param tf.tensor x_val: input tensor of the testing dataset
        :param tf.tensor y_val: output tensor of the testing dataset
        :param str metric: name of the error type, choose from 'MSE' or 'MAE'
        :return: tf.tensor mean_error: the mean squared/absolute error value,
        tf.tensor errors: the squared/absolute errors over inputs,
        tf.tensor y_pred: the prediction on the inputs of the testing dataset
        """

        y_pred = self.model.predict(x_val)
        errors = None
        if metric == 'MSE':
            errors = tf.square(y_val - y_pred)
        elif metric == 'MAE':
            errors = tf.abs(y_val - y_pred)

        mean_error = tf.reduce_mean(errors)

        return mean_error, errors, y_pred

    def prediction_time(self, batch_size, executions=1000):
        """
        Helper function to measure the mean prediction time of the neural network.

        :param int batch_size: dummy batch size of the input tensor
        :param int executions: number of performed executions to determine the mean value
        :return: float mean_prediction_time: the mean prediction time of the neural network on all executions
        """
        X = tf.random.uniform(shape=[executions, batch_size, self.input_dim], dtype=self.dtype)

        start_time = time.time()
        for execution in range(executions):
            _ = self.predict(X[execution])
        prediction_time = time.time() - start_time
        mean_prediction_time = prediction_time / executions

        return mean_prediction_time
