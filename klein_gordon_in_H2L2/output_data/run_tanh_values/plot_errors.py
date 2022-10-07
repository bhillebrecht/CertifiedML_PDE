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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from helpers.csv_helpers import import_csv

import json
import numpy as np
import matplotlib.pyplot as plt

from helpers.nn_parametrization import get_param_as_float
from helpers.plotting import new_fig, save_fig


if __name__ == "__main__":

    all_files = os.listdir(os.path.dirname(__file__))

    # count available time points
    counter = 0
    for file in all_files:
        if file[-4:] == "json":    
            counter = counter+1

    # initialize data set
    data = import_csv(os.path.join(os.path.dirname(__file__),"run_tanh_kleingordon_ref_t_error_over_time.csv"), has_headers=True)
    data= data[:,1:]

    # plot data
    cmap = plt.get_cmap("RdBu")
    blue = cmap(1.0)
    lightblue = cmap(.8)
    red = cmap(0.0)
    lightred = cmap(0.2)

    sortindex = np.argsort(data[:, 0])
    data = data[sortindex, :]

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    ax.set( ylabel=r'$E(t)$')
    ax.set( xlabel="$t$")
    ax.set (yscale="log")

    ax.grid(True, which='both')
    ax.set(xlim=[np.min(data[:,0]), np.max(data[:,0])])

    ax.plot(data[:,0:1], data[:,1:2]+ data[:,2:3] +data[:,5:6], linewidth=2, color=blue, linestyle="-", label="$E_\mathrm{PI} + E_\mathrm{init} + E_\mathrm{BC}$")
    ax.plot(data[:,0:1], data[:,1:2], linewidth=2, color=blue, linestyle="--", label="$E_\mathrm{init}$")
    ax.plot(data[:,0:1], data[:,2:3], linewidth=2, color=blue, linestyle=":", label="$E_\mathrm{PI}$")
    #ax.plot(data[:,0:1], data[:,5:6], linewidth=2, color=blue, linestyle="-.", label="$E_\mathrm{BC}$")
    ax.plot(data[:,0:1], data[:,4:5], linewidth=2, color=lightblue, linestyle="-", label="$E_\mathrm{ref}$")

    ax.legend(loc='best')
    fig.tight_layout()

    plt.show()
    save_fig(fig, "all_errors_plotted_0_01", os.path.join(os.path.dirname(__file__),"..", "run_tanh_figures"))

    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    ax.set( ylabel=r'$N_{SI}(t)$')
    ax.set( xlabel="$t$")

    ax.grid(True, which='both')
    ax.set(xlim=[np.min(data[:,0]), np.max(data[:,0])])

    ax.plot(data[:,0:1], data[:,3:4], linewidth=2, color=blue, linestyle="--", label="$N_{SP}$")

    ax.legend(loc='best')
    fig.tight_layout()

    plt.show()
    save_fig(fig, "all_NSP_plotted_0_01", os.path.join(os.path.dirname(__file__),"..", "run_tanh_figures"))
    
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)

    ax.set( ylabel=r'$(E_\mathrm{ref}(t))/(E_\mathrm{PI}(t) + E_\mathrm{init}(t))$')
    ax.set( xlabel="$t$")

    ax.grid(True, which='both')
    ax.set(xlim=[np.min(data[:,0]), np.max(data[:,0])])

    ax.plot(data[1:,0:1], (data[1:, 4:5])/(data[1:,2:3]+data[1:,1:2]), linewidth=2, color=blue, linestyle="--")

    ax.legend(loc='best')
    fig.tight_layout()

    plt.show()
    save_fig(fig, "Eref_to_EPI", os.path.join(os.path.dirname(__file__),"..", "run_tanh_figures"))
