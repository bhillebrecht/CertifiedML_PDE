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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..","helpers"))

import numpy as np
import matplotlib.pyplot as plt

from csv_helpers import import_csv
from plotting import new_fig, save_fig


if __name__ == "__main__":

    t_seq = np.linspace(0, 0.2, 101)
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    cmap = plt.get_cmap("RdBu")
    colors = np.concatenate([cmap(np.linspace(1.0, 0.6, 16)),cmap(np.linspace(0, 0.4, 16)), cmap(np.linspace(0.6, 1.0, 16))])
    ls = [':', '-', '--', '-.', 
          '--',':', '-.', '-', 
          '-.','--', '-',':' ]
    ls = np.concatenate([ls,ls,ls,ls,ls,ls], axis = 0)
    
    # plot u
    index = 0
    subindex = 0
    for time in t_seq:
        data = import_csv(os.path.join(os.path.dirname(__file__), "run_tanh_kleingordon_ref_t"+"{:3.3f}".format(time)+"_simple.csv"), has_headers=True)

        ax.set( ylabel=r'$u(t,x)$')
        ax.set( xlabel="$x$")

        ax.grid(True, which='both')
        ax.set(xlim=[0,1])

        if index % 16 == 0:
            ax.plot(data[:,1:2], data[:,3:4], linewidth=1.5, color=colors[subindex], linestyle=ls[subindex], label=r'$t= $'+str(time))
            subindex = subindex +1
        elif index % 4 == 0:
            ax.plot(data[:,1:2], data[:,3:4], linewidth=1.5, color=colors[subindex], linestyle=ls[subindex])
            subindex = subindex +1
        index= index+1

    ax.legend(loc='best')
    fig.tight_layout()

    save_fig(fig, "timeseries_tanh_u", os.path.join(os.path.dirname(__file__), "..", "run_tanh_figures"))
    plt.show()

    # plot u
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
        
    index = 0
    subindex = 0
    for time in t_seq:
        data = import_csv(os.path.join(os.path.dirname(__file__), "run_tanh_kleingordon_ref_t"+"{:3.3f}".format(time)+"_simple.csv"), has_headers=True)

        ax.set( ylabel=r'$\partial_t u(t,x)$')
        ax.set( xlabel="$x$")

        ax.grid(True, which='both')
        ax.set(xlim=[0,1])

        if index % 16 == 0:
            ax.plot(data[:,1:2], data[:,2:3], linewidth=1.5, color=colors[subindex], linestyle=ls[subindex], label=r'$t= $'+str(time))
            subindex = subindex+1
        elif index% 4 == 0:
            ax.plot(data[:,1:2], data[:,2:3], linewidth=1.5, color=colors[subindex], linestyle=ls[subindex])
            subindex = subindex+1
        index= index+1

    ax.legend(loc='best')
    fig.tight_layout()

    save_fig(fig, "timeseries_tanh_dtu", os.path.join(os.path.dirname(__file__), "..", "run_tanh_figures"))
    plt.show()