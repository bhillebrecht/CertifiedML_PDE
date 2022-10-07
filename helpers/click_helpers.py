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
import logging
import click

from helpers.error_helpers import ConvertStringToErrorEstimationType, ErrorEstimationType
from helpers.error_nn import generate_error_training_data_and_train
from helpers.run import eval_pinn
from helpers.train import train_pinn
from helpers.extract import extract_kpis
from helpers.target_helpers import get_target_utilities, set_target, set_target_path

@click.group()
@click.option('-t', '--target', type=click.Choice(['elem', 'pendulum', 'user'], case_sensitive=False), required=True, help='Selects target, valid values are elem, pendulum or user')
@click.option('-u', '--user_file', required=False, help='Sets the user file to be imported and used during training/extraction and running the PINN. It must contain a collection of mandatory functions, as defined in _template_appl. The parent directory is used as base directory. It is expected to find both, input and output directories next to it as well as required config files.' )
@click.option('-ll', '--log_level', type=click.Choice(['info', 'warning', 'error']), default="info", help='Sets log level accordingly. Valid values are info, warning, error')
@click.pass_context
def click_helper_select(ctx, target, user_file, log_level):
    # set log level as specified in command (default == info)
    if log_level == 'info' :
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == 'warning':
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    if target == 'user' and user_file is None:
        logging.error("Base directory must be set in case of a user target")
        sys.exit()
    
    if user_file is not None: 
        set_target_path(os.path.abspath(user_file))

    # store target to import adequate libs and params in sub commands
    set_target(target)

@click_helper_select.command()
@click.option('-ae', required=True, type=click.Choice(['point', 'domain', 'none']), help="Choses type of a posteriori error computation: pointwise (suitable for all ODEs), domainwise (suitable for global qualities for PDEs) or disable error estimation completely (none)." )
@click.option('-i', '--input_file', required = False, help="Set input file on which the network shall be evaluated")
@click.option('-eps', '--epsilon', default=0.33, required=False, type=float, help="Set fraction of expected error in the machine learning prediction that might be added due to numerical integral computation")
@click.pass_context
def run(ctx, ae, input_file, epsilon):
    appl_path, create_fn, load_fn, _, _, pevco = get_target_utilities()
    eet = ConvertStringToErrorEstimationType(ae)
    if eet == ErrorEstimationType.INVALID:
        # this should never happen. It would be a dev error
        logging.error("Wrong error estimation type chosen")
        sys.exit()
    if eet == ErrorEstimationType.DOMAINWISE and input_file is None:
        logging.error("No input file may only be chosen for pointwise error estimation if the example allows it")
        sys.exit()        
    eval_pinn(create_fn, load_fn, input_file, appl_path, epsilon, eet, pevco)

@click_helper_select.command()
@click.option('-lw', is_flag=True, help='Load weights from predefined path (output_data/weights). This can only be used if data has been stored previously. Default = false')
@click.option('-nsr',  default=False,  help='Avoid storing results, by default, neither weights nor result parameters are stored')
@click.option('-i', '--input_file', required=False, type=click.STRING, help='Sets input file for training')
@click.pass_context
def train(ctx, lw, nsr, input_file):  
    # set target
    appl_path, create_fn, load_fn, ptco, _, _ = get_target_utilities()

    # set store results parameter
    sr = not nsr
    
    # train pinn
    train_pinn(create_fn, load_fn, input_file, lw, sr, appl_path, ptco)

@click_helper_select.command()
@click.option('--mu_factor', type=float, default=0.1, help='Sets the factor to multiply avg deviation of ODE with to smoothen upper limit on deviation.')
@click.option('-i', '--input_file', required = False, help="Determines initial values for PDE. This is then stored together with KPIs. Be aware: the implicit assumption is that the points determining the initial state are chosen equidistantly and can thus be integrated easily.")
@click.option('-ae', type=click.Choice(['point', 'domain']), default='point', help="Choses type of a posteriori error computation: pointwise (suitable for all ODEs), domainwise (suitable for global qualities for PDEs)." )
@click.pass_context
def extract(ctx, mu_factor, input_file, ae):
    appl_path, create_fn, load_fn, _, peco, _= get_target_utilities()
    if input_file is not None:
        extract_kpis(create_fn, appl_path, mu_factor, peco, ae, load_fun=load_fn, input_file=input_file )
    else:
        extract_kpis(create_fn, appl_path, mu_factor, peco, ae)

@click_helper_select.command()
@click.option('-i', '--input_file', required = False, help="Set input file on which the network shall be evaluated")
@click.option('-eps', '--epsilon', default=0.33, required=False, type=float, help="Set fraction of expected error in the machine learning prediction that might be added due to numerical integral computation")
@click.option('--reset', is_flag=True, help="If set to true, weights for error NN are reset")
@click.pass_context
def train_error_net(ctx, input_file, epsilon, reset):  
    appl_path, create_fn, _, _, _, _ = get_target_utilities()
    # train pinn
    generate_error_training_data_and_train(appl_path, create_fn, epsilon, input_file, not reset)