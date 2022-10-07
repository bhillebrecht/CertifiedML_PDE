import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "helpers"))

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
    n_x = 101
    n_y = 101

    #alltimes = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    alltimes = np.linspace(0,1.0,21)

    for time in alltimes:
        X, Y = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
        t = np.ones((n_x*n_y, 1))*time

        u = -np.cos(X)*np.sin(Y)*np.exp(-2*time)
        v = np.sin(X)*np.cos(Y)*np.exp(-2*time)
        p = -1/4*(np.cos(2*X)+ np.cos(2*Y))*np.exp(-4*time)

        export_csv(np.concatenate((t, np.reshape(X, (n_x*n_y, 1)), np.reshape(Y, (n_x*n_y,1)), np.reshape(u, (n_x*n_y, 1)), np.reshape(v, (n_x*n_y,1)), np.reshape(p, (n_x*n_y,1))), axis=1), os.path.join(os.path.dirname(__file__), "taylor_t"+"{:3.2f}".format(time)+".csv"))
