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

if __name__ == "__main__":

    n_x = 201

    x_domain = np.linspace(0,1, n_x)
    x_domain = np.reshape(x_domain, (n_x, 1))

    f_0 = np.cos(2*np.pi*x_domain)
    f_1 = 0.5*np.cos(4*np.pi*x_domain) #+ np.cos(8*np.pi*x_domain)

    export_csv(np.concatenate((np.zeros((n_x,1)), x_domain , f_1, f_0) , axis=1), os.path.join(os.path.dirname(__file__), "kleingordon_input_simple.csv"), columnheaders=np.array(['t', 'x', 'dtu', 'u']))
