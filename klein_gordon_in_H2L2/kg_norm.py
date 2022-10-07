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

def compute_kg_single_point_norm_contribution(f, outdim= None):
    """
    Computes contribution of a single value to the extended H2 norm for KG equation

    param np.array or tf.tensor f : array over which the H2 norm shall be computed
    """
    # assert correct parameter hand over
    if outdim is not None:
        if not isinstance(outdim, int):
            logging.error("Coding Error: the number of relevant output dimensions must be integer")
            sys.exit() 

    # reduce output dimensions if necessary
    if outdim is not None:
        logging.error("The output dimension may not be reduced for kg norm")
        sys.exit() 

    val = tf.square(f)
    scaling = tf.convert_to_tensor(np.array([1, 0.25]))
    val = tf.math.multiply(val, scaling)

    # reduce dimensions 
    val =  tf.reduce_sum( val , axis=1)

    return val

def compute_kg_norm_laplace_contribution(f, x):
    """
    Computes the contribution of the Laplace operator/the gradient to the equivalent H2 norm
    """
    if x.ndim > 1 and x.shape[1] > 1:
        logging.error("The kg norm may only be used in 1D")
        sys.exit() 

    lap = np.diff(f[:, 1], 1)/(x[1,0]- x[0,0])
    lI1, lI2, lE = compute_1d_integral(np.square(lap), x[1:-1,:])

    return np.sqrt(lI1), np.sqrt(lI2), np.sqrt(lE)

def compute_kg_norm_from_contributions(single_point_contribs, x):
    """
    Computes KG norm equivalent to H2 norm

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

def compute_kg_norm(f, x, outdim=None):
    """
    Computes KG norm

    param np.array or tf.tensor f : array over which the KG norm shall be computed
    param np.array x : underlying space grid 
    param int outdim : will be ignored.
    """
    # sum over relevant output dimensions
    spc = compute_kg_single_point_norm_contribution(f)
    I1, I2, E = compute_kg_norm_from_contributions(spc, x)

    # compute laplace contribution
    lI1, lI2, lE = compute_kg_norm_laplace_contribution(f, x)

    # return
    return np.sqrt(I1**2+lI1**2), np.sqrt(I2**2+lI2**2), np.sqrt(E**2+lE**2)