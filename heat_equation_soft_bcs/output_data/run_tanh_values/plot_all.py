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
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..","helpers"))
from csv_helpers import import_csv
from plotting import new_fig, save_fig


if __name__ == "__main__":

    t_seq = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.get_cmap("RdBu")
    colors = np.concatenate([cmap(np.linspace(1.0, 0.6, 5)),cmap(np.linspace(0, 0.4, 5)), cmap(np.linspace(0.6, 1.0, 5))])
    ls = [':', '-', '--', '-.', 
          '--',':', '-.', '-', 
          '-.','--', '-',':' ]
    index = 0
    for time in t_seq:
        data = import_csv(os.path.join(os.path.dirname(__file__), "run_tanh_test_data_t"+"{:3.2f}".format(time)+".csv"), has_headers=True)

        ax.set( ylabel=r'$u(t,x)$')
        ax.set( xlabel="$x$")

        ax.grid(True, which='both')
        ax.set(xlim=[0,1])

        ax.plot(data[:,1:2], data[:,2:3], linewidth=1.5, color=colors[index], linestyle=ls[index], label=r'$t= $'+str(time))
        index= index+1

    ax.legend(loc='best')
    fig.tight_layout()

    save_fig(fig, "timeseries_tanh", os.path.join(os.path.dirname(__file__), "..", "run_tanh_figures"))
    plt.show()