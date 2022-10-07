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


import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "helpers"))
from csv_helpers import export_csv

def startcondition(x):
    f = np.zeros((x.shape[0], 1))
    for iter in range(x.shape[0]):
        if np.abs(x[iter, 0])+ np.abs(x[iter, 1]) > 0.5 :
            f[iter] = 0
        else:
            f[iter] = 0.5 - np.abs(x[iter,0]) - np.abs(x[iter, 1])
    return f

if __name__ == "__main__":
    ## number of test points
    n_x = 201
    n_y = 201

    x0 = np.reshape(np.linspace(-2,2,n_x), (n_x,1)) 
    x0 = np.repeat(x0, n_y)
    x0 = np.reshape(x0, (n_x*n_y, 1))
    
    y0 = np.reshape(np.linspace(-2,2,n_y), (n_y,1)) 
    y0 = np.repeat(y0, n_x)
    y0 = np.reshape(y0, (n_y, n_x))
    y0 = y0.transpose()
    y0 = np.reshape(y0, (n_x*n_y, 1))

    t = np.zeros(x0.shape)
    f = startcondition(np.concatenate([x0, y0], axis = 1))

    export_csv(np.concatenate((t, x0, y0, f), axis=1), os.path.join(os.path.dirname(__file__), "hat_input.csv"))
