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

import logging
import pandas
import numpy as np
import os

def export_csv(data, filename, columnheaders=None, rowheaders=None) -> None:
    """
    Helper function to export data as a csv including optional column- and rowheaders
    
    :param np.array data           data to be exported (2D array)
    :param string   filename       absolute path to target file
    :param np.array columnheaders  name of columns, length of array must agree to data.shape[1]
    :param np.array rowheaders     name of rows, length of array must agree to data.shape[0]
    """
    if rowheaders is not None:
        if data.shape[0]!=rowheaders.shape[0]:
            logging.error("Number of rowheaders does not agree with number of rows in provided data")
            return
        else: 
            rowheaders = np.array([rowheaders])
    if columnheaders is not None:
        if data.shape[1] != columnheaders.shape[0]:
            logging.error("Number of columnheaders does not agree with number of columns in provided data")
            return
        else: 
            columnheaders = np.array([columnheaders])
    
    formatteddata = data# np.reshape(np.array(["{%.2e}"% x for x in data.reshape(data.size)]), data.shape)
    if rowheaders is not None:
        formatteddata = np.concatenate((np.transpose(rowheaders), formatteddata), axis=1)
    if columnheaders is not None:
        if rowheaders is not None:
            columnheaders = np.concatenate(([[""]], columnheaders), axis=1)
        formatteddata = np.concatenate((columnheaders, formatteddata), axis=0)

    df = pandas.DataFrame(formatteddata)
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    df.to_csv(filename, mode="w+", index=False, header=False)

def import_csv(filename, has_headers=True) -> np.array:
    """ 
    Helper function to import data from a csv file. Return is the imported 

    :param   string filename        path to the target file
    :param   boolean has_headers    indicates if csv file has headers, default is true
    :return  np.array               np.array containing the content of the csv file
    """
    if has_headers:
        df = pandas.read_csv(filename)
    else: 
        df = pandas.read_csv(filename, header=None)
    return df.to_numpy()
