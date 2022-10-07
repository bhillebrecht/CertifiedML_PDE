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

import numpy as np
import tensorflow as tf

class BoundaryCondition:
    def __init__(self, i, o, c, r, ofs=None):
        self.INPUT_INDICES = i
        self.OUTPUT_INDICES = o
        self.CENTER = c
        self.RADIUS = r
        self.OffsetLayer = ofs

class LengthFactor(tf.keras.layers.Layer):
    # currently this can only set all 

    def __init__(self, input_dim, units, ccenter, radius, ind) -> None:
        super(LengthFactor, self).__init__(dtype='float64')

        self.domain_center = ccenter
        self.radius = radius
        self.ind = ind
        self.units = units
        self.input_dim = input_dim

    def call(self, inputs):
        output = tf.ones((tf.shape(inputs)[0], 1), dtype="float64")
        for index in range(0, self.ind.shape[0]):
            output = tf.multiply(output,  (self.radius[index]**2 - (tf.reshape(inputs[:,self.ind[index]], (tf.shape(inputs)[0], 1)) - self.domain_center[index])**2 ))
        output = tf.repeat(output, self.units, axis=1)
        return output

class MultiplyLengthFactor(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, oi) -> None:
        super(MultiplyLengthFactor, self).__init__(dtype='float64')
      
        self.units = units
        self.input_dim = input_dim
        self.oi = oi

    def call(self, inputs):
        listoftensors = []
        for index in range(0, self.units):
            if index in self.oi:
                listoftensors.append(tf.multiply(inputs[:, index], inputs[:, self.units+self.oi[np.where(self.oi == index)[0][0]]]))
            else:
                listoftensors.append(inputs[:, index])
        output = tf.stack(listoftensors, axis=1)
        return output        
        # return  10.0*tf.math.multiply(tf.reshape(inputs[:,0], (tf.shape(inputs)[0], 1)), tf.reshape(inputs[:,1], (tf.shape(inputs)[0], 1)))


class AddOffsetLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, oi) -> None:
        super(AddOffsetLayer, self).__init__(dtype='float64')

        self.units = units
        self.input_dim = input_dim
        self.oi = oi

    def call(self, inputs):
        listoftensors = []
        for index in range(0, self.units):
            if index in self.oi:
                listoftensors.append(inputs[:, index] + inputs[:, self.units+self.oi[np.where(self.oi == index)[0][0]]] )
            else:
                listoftensors.append(inputs[:, index])
        output = tf.stack(listoftensors, axis=1)
        return output  


