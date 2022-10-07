import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "helpers"))

import numpy as np
import matplotlib.pyplot as plt

from csv_helpers import export_csv
from plotting import new_fig, save_fig
from input_generator import generate_collocation_points

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
    n_x = 31
    n_y = 31

    x_lim_low = 0
    x_lim_high = np.pi
    y_lim_low = 0
    y_lim_high = np.pi

    X, Y = np.meshgrid(np.linspace(x_lim_low, x_lim_high, n_x), np.linspace(y_lim_low, y_lim_high, n_y))
    t = np.zeros((n_x*n_y, 1))

    u = -np.cos(X)*np.sin(Y)
    v = np.sin(X)*np.cos(Y)
    p = -1/4*(np.cos(2*X)+ np.cos(2*Y))

    export_csv(np.concatenate((t, np.reshape(X, (n_x*n_y, 1)), np.reshape(Y, (n_x*n_y,1)), np.reshape(u, (n_x*n_y, 1)), np.reshape(v, (n_x*n_y,1)), np.reshape(p, (n_x*n_y,1))), axis=1), os.path.join(os.path.dirname(__file__), "taylor_input.csv"))

    ## plot u component
    fig = new_fig(ratio=0.75)
    ax = fig.add_subplot(111)

    ax.set( ylabel="$y$")
    ax.set( xlabel="$x$")
    ax.grid(False, which='both')

    c = ax.pcolormesh(np.reshape(X, (n_x, n_y)), np.reshape(Y, (n_x, n_y)), np.reshape(u, (n_x, n_y)), cmap='RdBu', label="input")
    ax.axis([x_lim_low, x_lim_high, y_lim_low, y_lim_high])
    ax.set_aspect(1)
    fig.colorbar(c, ax=ax)

    plt.show()
    save_fig(fig, "taylor_input_u" , os.path.dirname(__file__))

    ## plot v component
    fig = new_fig(ratio=0.75)
    ax = fig.add_subplot(111)

    ax.set( ylabel="$y$")
    ax.set( xlabel="$x$")
    ax.grid(False, which='both')

    c = ax.pcolormesh(np.reshape(X, (n_x, n_y)), np.reshape(Y, (n_x, n_y)), np.reshape(v, (n_x, n_y)), cmap='RdBu',  label="input")
    ax.axis([x_lim_low, x_lim_high, y_lim_low, y_lim_high])
    ax.set_aspect(1)
    fig.colorbar(c, ax=ax)

    plt.show()
    save_fig(fig, "taylor_input_v" , os.path.dirname(__file__))

    ## plot p component
    fig = new_fig(ratio=0.75)
    ax = fig.add_subplot(111)

    ax.set( ylabel="$y$")
    ax.set( xlabel="$x$")
    ax.grid(False, which='both')

    c = ax.pcolormesh(np.reshape(X, (n_x, n_y)), np.reshape(Y, (n_x, n_y)), np.reshape(p, (n_x, n_y)), cmap='RdBu', label="input")
    ax.axis([x_lim_low, x_lim_high, y_lim_low, y_lim_high])
    ax.set_aspect(1)
    fig.colorbar(c, ax=ax)

    plt.show()
    save_fig(fig, "taylor_input_p" , os.path.dirname(__file__))