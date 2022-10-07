###################################################################################################
# Copyright (c) 2021 Jonas Nicodemus
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
#
# This file incorporates work and modifications to the originally published code
# according to the previous license by the following contributors under the following licenses
#
#   Copyright (c) 2022 Birgit Hillebrecht
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
###################################################################################################


import logging
import os

import matplotlib.pyplot as plt
import numpy as np

def fig_size(width_pt, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Returns the width and heights in inches for a matplotlib figure.

    :param float width_pt: document width in points, in latex can be determined with \showthe\linewidth
    :param float fraction: fraction of the width with which the figure will occupy
    :param float ratio: ratio of the figure, default is the golden ratio
    :param tuple subplots: the shape of subplots
    :return: float fig_width_in: width in inches of the figure, float fig_height_in: height in inches of the figure
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def new_fig(width_pt=410, fraction=1, ratio=(5 ** .5 - 1) / 2, subplots=(1, 1)):
    """
    Creates new instance of a `matplotlib.pyplot.figure` fig by using the `fig_size` function.

    :param float width_pt: document width in points, in latex can be determined with \showthe\textwidth
    :param float fraction: fraction of the width with which the figure will occupy
    :param float ratio: ratio of the figure, default is the golden ratio
    :param tuple subplots: the shape of subplots
    :return: matplotlib.pyplot.figure fig: instance of a `matplotlib.pyplot.figure` with desired width and height
    """
    fig = plt.figure(figsize=fig_size(width_pt, fraction, ratio, subplots))
    return fig


def save_fig(fig, name, path, tight_layout=True):
    """
    Saves a `matplotlib.pyplot.figure` as pdf file.

    :param matplotlib.pyplot.figure fig: instance of a `matplotlib.pyplot.figure` to save
    :param str name: filename without extension
    :param str path: path where the figure is saved, if None the figure is saved at the results directory
    :param bool crop: bool if the figure is cropped before saving
    """
    if tight_layout:
        fig.tight_layout()

    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(os.path.join(path, f'{name}.pdf'), transparent=True)

def plot_states_1D(T, Z_ref, name, filepath, ylabeli=r'$x(t)$', legend_ref =r'$x(t)$', legend_pred=r'$\hat{x}(t)$', Z_pred=None, Z_mpc=None):
    linewidth = 2
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$', ylabel=ylabeli)
    ax.set(xlim=[np.min(T), np.max(T)])
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    ax.plot(T, Z_ref, linewidth=linewidth, label=legend_ref, c=colors[2])
    if Z_pred is not None:
        ax.plot(T, Z_pred, linestyle='--', linewidth=linewidth, label=legend_pred,
                    c=colors[1])

    ax.grid('on')
    ax.legend(loc='best')
    fig.tight_layout()
    if filepath is not None:
        save_fig(fig, name, filepath)
    plt.show()

def plot_trustzone(T, Z, Z_err, Z_ref,name, filepath):
    Z_err = np.reshape(Z_err, Z.shape)
    linewidth = 2
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$', ylabel=r'x(t)')
    ax.set(xlim=[np.min(T), np.max(T)])
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    ax.plot(T, Z, linewidth=linewidth, c=colors[2], label=r'$\hat{x}(t)$')
    ax.plot(T, Z_ref, linewidth=linewidth, linestyle='--', c=colors[0], label=r'$x(t)$')
    ax.plot(T, Z-Z_err, linestyle='--', linewidth=linewidth, c=colors[1])
    ax.plot(T, Z+Z_err, linestyle='--', linewidth=linewidth, c=colors[1])

    ax.grid('on')
    ax.legend(loc='best')
    fig.tight_layout()
    save_fig(fig, name, filepath)
    plt.show()

def plot_absolute_errors(T, Z_pred_err, name, filepath, Z_ref=None,  Z_pred=None):
    colors = ['tab:orange', 'tab:red', 'tab:green', 'tab:blue']
    fig = new_fig()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='Time $t$ ', ylabel=r'Error')
    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set(xlim=[np.min(T), np.max(T)])

    ax.plot(T, Z_pred_err[:,0]+Z_pred_err[:,1], linewidth=2, color= colors[0], label=r'$E_\mathrm{Init} + E_\mathrm{PI}$')
    ax.plot(T, Z_pred_err[:,0], linewidth=2, linestyle="--",  color= colors[3], label=r'$E_\mathrm{Init}$')
    ax.plot(T, Z_pred_err[:,1], linewidth=2, linestyle="--", color= colors[2], label=r'$E_\mathrm{PI}$')

    if Z_pred is not None and Z_ref is not None:
        logging.info(f'{Z_ref - Z_pred}')
        if  Z_ref.ndim>1 :
            abs_errors = (np.sum((Z_ref - Z_pred)**2, axis=1))**0.5
        else : 
            abs_errors = np.absolute(Z_ref-Z_pred)
        ax.plot(T, abs_errors, linewidth=2, color= colors[1], label=r'$\|\hat{x}(t)-x(t)|$')

    ax.legend(loc='best')
    save_fig(fig, name, filepath)
    fig.tight_layout()
    plt.show()

def plot_loss(T, filename, filepath, xname, yname, Loss1, Loss2=None, Loss3= None, Loss4= None) :
    t_sorted = np.argsort(T[:,0])
    if Loss1.ndim > 1:
        Loss1 = Loss1[t_sorted, 0]
    else:
        Loss1 = Loss1[t_sorted]
    T_sort = T[t_sorted]

    axxmax = 1
    axymax = 1
    if Loss2 is not None:
        axxmax = 2
    if Loss3 is not None or Loss4 is not None:
        axymax = 2

    colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue']
    fig = new_fig()
    ax = fig.add_subplot(axxmax, axymax, 1)
    ax.set( ylabel=yname)
    # ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.set(xlim=[np.min(T), np.max(T)])

    ax.plot(T_sort, Loss1, linewidth=2,c=colors[0])

    if Loss2 is not None: 
        Loss2 = Loss2[t_sorted]
        ax2= fig.add_subplot(axxmax,axymax,2)
        ax2.set(xlim=[np.min(T), np.max(T)])
        ax2.plot(T_sort, Loss2, linewidth=2, c=colors[1])
        if Loss4 is None: 
            ax2.set(xlabel=xname)
    else:
        ax.set(xlabel=xname)

    if Loss3 is not None: 
        Loss3 = Loss3[t_sorted]
        ax3= fig.add_subplot(axxmax,axymax,3)
        ax3.set(xlim=[np.min(T), np.max(T)])
        ax3.plot(T_sort, Loss3, linewidth=2, c=colors[2])
        ax3.set(xlabel=xname)

    if Loss4 is not None:
        Loss4 = Loss4[t_sorted]
        ax4= fig.add_subplot(axxmax,axymax,4)
        ax4.set(xlim=[np.min(T), np.max(T)])
        ax4.plot(T_sort, Loss4, linewidth=2, c=colors[3])
        ax4.set(xlabel=xname)

    save_fig(fig, filename, filepath)
    plt.show()