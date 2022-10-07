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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..","..",  "helpers"))
from csv_helpers import export_csv
from plotting import new_fig, save_fig

def reference_values(x, t):
    f = np.zeros((x.shape[0], 1))
    for iter in range(x.shape[0]):
        x_per = x[iter, 0] + 0.2*t
        while x_per > 2:
            x_per = x_per -4
        while x_per <-2:
            x_per = x_per +4

        y_per = x[iter, 1] + 0.5*t
        while y_per > 2:
            y_per = y_per -4
        while y_per < -2:
            y_per = y_per +4

        if np.abs(x_per)+ np.abs(y_per) < 0.5:
            f[iter] = 0.5 - np.abs(x_per)- np.abs(y_per)
        else:
            f[iter] = 0
    return f


if __name__ == "__main__":

    doplot = False

    ## number of test points
    t_seq = np.linspace(0, 8, 81)

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

    for time in t_seq:
        t = np.ones(x0.shape)*time    
        f = reference_values(np.concatenate([x0, y0], axis = 1), time)
        export_csv(np.concatenate((t, x0, y0, f), axis=1), os.path.join(os.path.dirname(__file__), "input_eval_"+"{:3.2f}".format(time)+".csv"))

        if doplot:
            fig = new_fig()
            ax = fig.add_subplot(111)

            ax.set( ylabel="$y$")
            ax.set( xlabel="$x$")
            ax.grid(False, which='both')
            ax.set(xlim=[-2,2])
            ax.set(ylim=[-2,2])

            c = ax.pcolormesh(np.reshape(x0, (n_x, n_y)), np.reshape(y0, (n_x, n_y)), np.reshape(f, (n_x, n_y)), cmap='Blues', vmin=0, vmax=0.5)
            ax.axis([-2,2,-2,2])
            fig.colorbar(c, ax=ax)

            # plt.show()
            save_fig(fig, "input_eval_"+"{:3.2f}".format(time) , os.path.dirname(__file__))