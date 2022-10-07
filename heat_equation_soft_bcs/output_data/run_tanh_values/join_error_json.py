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
from helpers.nn_parametrization import get_param_as_float, has_param

if __name__ == "__main__":

    all_files = os.listdir(os.path.dirname(__file__))
    prefix =  "run_tanh_test_data_t"
    postfix =  "_error.json"

    # count available time points
    counter = 0
    for file in all_files:
        if file[-4:] == "json":    
            counter = counter+1

    # initialize data set
    data = np.zeros((counter, 8), dtype="float32")
    index = 0

    # extract data and fill it to data array
    for file in all_files:
        if file[-4:] == "json":
            head, tail = os.path.split(file)
            time = tail.replace(prefix, "")
            time = time.replace(postfix, "")
            
            data[index, 0] = float(time)
            data[index, 1] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_init")
            data[index, 2] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_PI")
            data[index, 3] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "N_SP")
            data[index, 4] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_ref")

            
            if has_param(os.path.join(os.path.dirname(__file__), file), "E_bc"):
                data[index, 5] = get_param_as_float(os.path.join(os.path.dirname(__file__), file), "E_bc")

            data[index, 6] = data[index, 1] + data[index, 2]+ data[index, 5]
            data[index, 7] = (data[index, 1] + data[index, 2]+ data[index, 5]) /data[index, 4]

            index = index+1

    sortedidx = np.argsort(data[:,0])
    data = data[sortedidx,:]

    export_csv(data, os.path.join(os.path.dirname(__file__), prefix + "_error_over_time.csv"), columnheaders=np.array(["t", "E_init", "E_PI", "N_SP", "E_ref", "E_bc", "E_tot", "E_rel"]), rowheaders=np.linspace(1, data.shape[0], data.shape[0]))

