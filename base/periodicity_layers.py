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

from helpers.globals import get_activation_function
from math import pi

class PeriodicLayers:
    def __init__(self, i, m):
        self.index_and_n = i
        self.m = m

class Periodic_Base(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, period, ind, tdc) -> None:
        super(Periodic_Base, self).__init__(dtype='float64')

        self.units = int(units)
        self.input_dim = input_dim
        self.period = period
        self.ind = ind
        self.tdc = tdc.astype('int')

        val_omega = 2*pi/ period 
        self.omega = self.add_weight( name='omega',
            shape=(int(units), ), initializer=tf.constant_initializer(val_omega) , trainable=False, dtype='float64'
        )

    def call(self, inputs):
        dtdc = int((self.tdc[1]- self.tdc[0])/3)
        params_phi = inputs[:, self.tdc[0]          :(self.tdc[0]+ dtdc)]
        params_a =   inputs[:, (self.tdc[0]+dtdc)   :(self.tdc[0]+ 2*dtdc)]
        params_c =   inputs[:, (self.tdc[0]+ 2*dtdc): self.tdc[1]]

        return get_activation_function()( tf.math.cos(self.omega* tf.reshape(inputs[:,self.ind], (tf.shape(inputs)[0], 1)) + params_phi) * params_a  + params_c)

    def get_config(self):
        config = super(Periodic_Base, self).get_config()
        config.update({"units": self.units})
        config.update({"input_dim": self.input_dim})
        config.update({"period": self.period})
        config.update({"ind": self.ind})
        config.update({"tdc": self.tdc})
        return config

class NonPeriodic_Base(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, ind) -> None:
        super(NonPeriodic_Base, self).__init__(dtype='float64')

        self.units = units
        self.input_dim = input_dim
        self.ind = ind

        # outer activation
        self.a = self.add_weight( name='a',
            shape=(1, int(units)), initializer="glorot_normal", trainable=True, dtype='float64'
        )
        self.c = self.add_weight( name='c',
            shape=(int(units), ), initializer="glorot_normal", trainable=True, dtype='float64'
        )

    def call(self, inputs):
        return get_activation_function()( tf.matmul( tf.reshape(inputs[:,self.ind], (tf.shape(inputs)[0], 1)) , self.a) + self.c)

    def get_config(self):
        config = super(NonPeriodic_Base, self).get_config()
        config.update({"units": self.units})
        config.update({"input_dim": self.input_dim})
        config.update({"ind": self.ind})
        return config

