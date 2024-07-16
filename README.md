# Certified Physics Informed Neural Networks

The code provides a standard mean to use certified PINNs for your system, which is goverened by an ODE or PDE. 

## Citation

Before using the here published code, be aware that the code is published under MIT common license, as found in the LICENSE file enclosed. To quote this work in publications, refer to the CITATION file, therein we kindly ask to consider citing

**Hillebrecht, B., & Unger, B. (2023). Rigorous a posteriori error bounds for PDE defined PINNs. IEEE Transactions on Neural Networks and Learning Systems, 1–11. https://doi.org/10.1109/TNNLS.2023.3335837**

upon use of this code to generate published results. 

The parts of the presented code, which cover ODEs, have been presented previously [Link to CertifiedML-ODE Repository](https://github.com/bhillebrecht/CertifiedML-ODE) as auxiliary material for 
- B. Hillebrecht and B. Unger, "[Certified machine learning: A posteriori error estimation for physics-informed neural networks](https://ieeexplore.ieee.org/document/9892569)", 2022 International Joint Conference on Neural Networks (IJCNN), 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892569.


## System Requirements

Install python 3.9.x, tensorflow 2.11.0 is still incompatible to python 3.10.x.
Use pip to install all packages listed in requirements.txt
```
pip install -r requirements.txt
```

## Configuration 

The two examples which were published in the above mentioned application are included as well. The here published code has two means to be configured and controlled:
- static parameters of the neural network and the training are set via configuration files
- commands and commandline options help control the neural networks and the current actions applied to those

### Configuration Files

Static configuration of the neural network and the training is done via .json files. One for the neural network in general
- "config_nn.json" with parameters as listed in ./json_schema_input/conig_nn_schema.json

and one for the training process steps
- in case only one step is needed "config_training.json" with mandatory parameters as listed in ./json_schema_input/config_training_schema.json (i.e. set the parameter "$schema" in the json file accordingly)
- multistep configurations "config_training.json" are required to fulfill ./json_schema_input/config_training_multistep_schema.json (i.e. set the parameter "$schema" in the json file accordingly)


### Command Line Configuration

Using commands, subcommands and command line options, you can control the framework. To find a full description of available options and commands, use

```
python .\certified_pinn.py --help
```

The central control unit has four subcommands, which depend on the results of the previous one
1. train
2. extract
3. either 
    1. run or
    2. train-error-net (only for pointwise errors)

In consequence, extract can not be called without training the NN before and run and train-error-net can not be called without extracting key parameters before using the command extract.

The target system is defined with option "-t", all examples beyond the ODE examples elem and pendulum are accessible via the user option, e.g.
```
python .\certified_pinn.py -t user -u heat_equation/heat_equation.py  train -i heat_equation/input_data/initial_data.csv
```

To find detailled information on the options of the commands use e.g. for "train"
```
python .\certified_pinn.py -t user --user  train --help
```
