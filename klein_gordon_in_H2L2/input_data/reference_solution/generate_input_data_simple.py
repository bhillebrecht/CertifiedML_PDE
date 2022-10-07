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
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..","..", "helpers"))
from csv_helpers import export_csv 
from plotting import new_fig, save_fig


if __name__ == "__main__":

    create_figures = False

    t = np.linspace(0, 0.2, 101)
    n_x = 201
    x_domain = np.linspace(0,1, n_x)
    x_domain = np.reshape(x_domain, (n_x, 1))

    a = 1.0
    b = 1.0/4.0
    lambda2 = np.pi * 2
    lambda4 = np.pi * 4
    #lambda8 = np.pi * 8


    for time in t:
        u_0 = np.cos(lambda2*x_domain)*np.cos(time*np.sqrt(lambda2**2*a**2+b)) \
            + 0.5* np.cos(lambda4*x_domain)*np.sin(time*np.sqrt(a**2*lambda4**2+b))/np.sqrt(a**2*lambda4**2+b)              
        u_1 = -np.cos(lambda2*x_domain)*np.sin(time*np.sqrt(lambda2**2*a**2+b))*np.sqrt(lambda2**2*a**2+b) \
            + 0.5*np.cos(lambda4*x_domain)*np.cos(time*np.sqrt(a**2*lambda4**2+b))
        export_csv(np.concatenate((np.ones((n_x,1))*time, x_domain , u_1, u_0) , axis=1), os.path.join(os.path.dirname(__file__), "kleingordon_ref_t"+"{:3.3f}".format(time)+"_simple.csv"), columnheaders=np.array(["t", "x", "dtu", "u"]))

        if create_figures: 
            fig = new_fig()
            ax = fig.add_subplot(1, 1, 1)

            colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
            ax.set( ylabel=r'Initial Conditions')
            ax.set( xlabel="$x$")

            ax.grid(True, which='both')
            ax.set(xlim=[0,1])

            ax.plot(x_domain[:,0], u_0, color=colors[0], linewidth=1.5, label="$u(t,x)$")
            ax.plot(x_domain[:,0], u_1, color=colors[1], linewidth=1.5, label="$\\partial_t u(t,x)$")

            ax.legend(loc='best')
            fig.tight_layout()

            save_fig(fig, "kleingordon_ref_t"+"{:3.2f}".format(time), os.path.dirname(__file__))
            plt.show()