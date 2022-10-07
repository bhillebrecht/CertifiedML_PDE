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
import importlib
from pathlib import Path

TARGET = None
TARGET_PATH = None

def get_target_utilities():
    global TARGET
    global TARGET_PATH
    # select target
    appl_path = os.path.dirname(__file__)
    if TARGET == 'elem':
        TARGET_PATH = os.path.join(os.path.dirname(__file__), '..', 'appl_elem', 'elem_pinn.py')

    elif TARGET == 'pendulum':
        TARGET_PATH = os.path.join(os.path.dirname(__file__), '..', 'appl_pendulum', 'inverse_pendulum.py')

    loader = importlib.machinery.SourceFileLoader(Path(TARGET_PATH).stem, TARGET_PATH)
    spec = importlib.util.spec_from_loader(Path(TARGET_PATH).stem, loader)
    mymodule = importlib.util.module_from_spec(spec)
    loader.exec_module(mymodule)

    create_pinn = mymodule.create_pinn
    load_data = mymodule.load_data
    post_train_callout = mymodule.post_train_callout
    post_extract_callout = mymodule.post_extract_callout
    post_eval_callout = mymodule.post_eval_callout
    appl_path = os.path.dirname(TARGET_PATH)

    return appl_path, create_pinn, load_data, post_train_callout, post_extract_callout, post_eval_callout

def get_target():
    global TARGET
    return TARGET

def get_target_path():
    global TARGET_PATH
    return TARGET_PATH

def set_target(str):
    global TARGET
    TARGET = str
    return

def set_target_path(str):
    global TARGET_PATH
    TARGET_PATH = str
    return