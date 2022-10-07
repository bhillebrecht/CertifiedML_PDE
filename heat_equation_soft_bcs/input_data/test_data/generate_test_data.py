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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from helpers.csv_helpers import export_csv
from helpers.plotting import new_fig, save_fig


if __name__ == "__main__":
    
    num_int = 200

    te = np.linspace(0, 0.5, 51)

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$', ylabel=r'u(t)')
    ax.set(xlim=[0,1])
    ax.set(ylim=[-1.1,1.1])
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']


    for time in te:
        t = np.ones((num_int+1, 1))*time
        x = np.reshape(np.linspace(0,1, num_int+1), t.shape)
        fx = np.sin(2 * np.pi * x)*np.exp(-0.2*4*np.pi**2*time)

        export_csv(np.concatenate((t, x, fx), axis=1), os.path.join(os.path.dirname(__file__), "test_data_t"+"{:3.2f}".format(time)+".csv"))
        ax.plot(x, fx, linewidth=1.5, c=colors[2], label=r'$t=$'+ "{:.4}".format(str(time)))
   
    ax.grid('on')
    ax.legend(loc='best')
    fig.tight_layout()
    save_fig(fig, "test_data", os.path.dirname(__file__))
