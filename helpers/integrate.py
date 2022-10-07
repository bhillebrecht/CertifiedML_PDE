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


def compute_integral_trpz(Y, dx, ndim = 1, boundary_factors = np.empty((0,0))):
    """
    Computes integral over Y using trapezoidal rule and step size dx

    :param np.array Y : values to integrate numerically
    :param float dx : step size in numerical integration
    """
    if not np.any(boundary_factors) and ndim != 1:
        logging.warning("Boundary factors can not be reasonably set to irrelevant for multidimensional integrals")

    integralvalue = 0
    if ndim == 1:
        integralvalue = dx/(2.0)*(np.sum(Y[0:(len(Y)-1)])+np.sum(Y[1:len(Y)]))
    else: 
        integralvalue = dx*np.sum(Y/(boundary_factors))
    return integralvalue

def determine_boundaryfactors_2dim(x, y, x_unique, y_unique): 
    bf = np.ones(x.shape)*4

    xmin, xmax = np.min(x_unique), np.max(x_unique)
    ymin, ymax = np.min(y_unique), np.max(y_unique)
    bx = np.concatenate( (np.where(x == xmin),np.where(x == xmax)))
    by = np.concatenate( (np.where(y == ymin),np.where(y == ymax)))

    bf[bx] = bf[bx]/2.0
    bf[by] = bf[by]/2.0

    bf = np.reshape(bf,(bf.shape[0],1))

    return bf

def splitgrid(x, y, x_unique, y_unique):
    x_space_by_half = x_unique[::2]
    y_space_by_half = y_unique[::2]

    indices = np.zeros((x_space_by_half.shape[0]*y_space_by_half.shape[0],))
    ci = 0
    for index in range(0, x.shape[0]):
        itemfoundx = np.where((x[index] == x_space_by_half))
        itemfoundy = np.where((y[index] == y_space_by_half))
        if itemfoundx[0].shape[0]> 0 and itemfoundy[0].shape[0]>0:
            indices[ci] = index
            ci = ci+1
    x2 = x[indices.astype('int')]
    y2 = y[indices.astype('int')]

    return x2, y2, x_space_by_half, y_space_by_half, indices

def compute_1d_integral(f, x):
    if not isinstance(f, np.ndarray):
        logging.error("Coding error. It is only allowed to hand numpy arrays to compute_1d_integral.")
        sys.exit()

    if f.ndim > 1:
        logging.error("Coding error. It is only allowed to hand 1d arrays as functions over 1 space dimension to compute_1d_integral.")
        sys.exit()

    # compute 2D integral with two spacings
    dx = x[1,0]- x[0,0]
    I1 = compute_integral_trpz(f, dx)
    I2, E = None, None
    if f.shape[0] > 1:
        I2 = compute_integral_trpz(f[::2], 2*dx)
        E = 0.8*np.abs(I1-I2)
    return I1, I2, E

def compute_2d_integral(f, x):
    if not isinstance(f, np.ndarray):
        logging.error("Coding error. It is only allowed to hand numpy arrays to compute_2d_integral.")
        sys.exit()

    if f.ndim > 1:
        logging.error("Coding error. It is only allowed to hand 1d arrays as functions over 2 space dimensions to compute_2d_integral.")
        sys.exit()

    # make sure that f is not a vector
    f = np.reshape(f, (f.shape[0],1))

    # get grid
    x_space = np.sort(np.unique(x[:,0]))
    y_space = np.sort(np.unique(x[:,1]))

    # Compute value of integral for 4N support intervals 
    bf = determine_boundaryfactors_2dim(x[:,0], x[:,1], x_space, y_space)
    I1 = compute_integral_trpz(f, (x_space[1]- x_space[0])*(y_space[1]-y_space[0]), ndim =2, boundary_factors=bf) 

    # Compute value of integral for N support intervals 
    x2, y2, x_space_by_half, y_space_by_half, indices = splitgrid(x[:,0], x[:,1], x_space, y_space)
    bf2 = determine_boundaryfactors_2dim(x2, y2, x_space_by_half, y_space_by_half)
    I2 = compute_integral_trpz(f[indices.astype('int')], (x_space_by_half[1]-x_space_by_half[0])*(y_space_by_half[1]-y_space_by_half[0]), ndim=2, boundary_factors=bf2)

    # Determine error 
    E = 16/17*np.abs(I2-I1)

    return I1, I2, E
