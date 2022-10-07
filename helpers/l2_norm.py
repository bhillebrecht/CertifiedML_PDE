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

import logging
import sys
import numpy as np
import tensorflow as tf

from helpers.integrate import compute_1d_integral, compute_2d_integral

def compute_l2_single_point_norm_contribution(f, outdim= None):
    """
    Computes contribution of a single value to the L2 norm

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    """
    # assert correct parameter hand over
    if outdim is not None:
        if not isinstance(outdim, int):
            logging.error("Coding Error: the number of relevant output dimensions must be integer")
            sys.exit() 

    # reduce output dimensions if necessary
    if outdim is not None:
        f = f[:,:outdim]

    val = tf.square(f)

    # reduce dimensions if necessary
    if tf.rank(val) >1:
        val =  tf.reduce_sum( val , axis=1)

    return val

def compute_l2_norm_from_contributions(single_point_contribs, x):
    """
    Computes L2 norm

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    param np.array x : underlying space grid 
    param int outdim : number of output dimesnions relevant for the L2 norm. Consider f[:, :outdim] for L2 norm only
    """
    # convert tf to np if necessary
    if tf.is_tensor(single_point_contribs):
        single_point_contribs = single_point_contribs.numpy()
    if tf.is_tensor(x):
        x = x.numpy()

    # compute integral 
    if x.shape[1] == 1:
        I1, I2, E = compute_1d_integral(single_point_contribs, x)
    elif x.shape[1] == 2:
        I1, I2, E = compute_2d_integral(single_point_contribs, x)

    # return
    return np.sqrt(I1), np.sqrt(I2), np.sqrt(E)

def compute_l2_norm(f, x, outdim=None):
    """
    Computes L2 norm

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    param np.array x : underlying space grid 
    param int outdim : number of output dimesnions relevant for the L2 norm. Consider f[:, :outdim] for L2 norm only
    """
    # sum over relevant output dimensions
    f = compute_l2_single_point_norm_contribution(f, outdim)
    I1, I2, E = compute_l2_norm_from_contributions(f, x)

    # return
    return I1, I2, E