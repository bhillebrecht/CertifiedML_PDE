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

class Taylor_BC_OffsetLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, units) -> None:
        super(Taylor_BC_OffsetLayer, self).__init__(dtype='float64')
        self.input_dim = input_dim
        self.units = units

    def call(self, inputs):
        y = tf.reshape(inputs[:,2], (tf.shape(inputs)[0], 1))
        x = tf.reshape(inputs[:,1], (tf.shape(inputs)[0], 1))

        pi2 = np.pi**2

        siny = tf.sin(y)
        sinx = tf.sin(x)
        exp2t = tf.exp(-2*tf.reshape(inputs[:,0], (tf.shape(inputs)[0], 1)))

        componentu = - (-(x - np.pi)**2 / pi2 + 1 ) + \
                     (-(x    )**2 / pi2 + 1 ) 
        componentu = componentu * tf.multiply( -siny, exp2t)

        componentv = - (-(y -np.pi)**2 / pi2 + 1 ) + \
                     (-(y    )**2 / pi2 + 1 )
        componentv = componentv * tf.multiply( sinx,  exp2t)

        return tf.concat((componentu, componentv), axis=1)
