# Heat equation with soft boundary constraints

## Problem Description

The boundary and initial value problem under consideration is
$$\partial_t u(t,x) = \tfrac{1}{5} \partial_{xx} u(t,x),$$
$$u(0, x) = \sin(2\pi x ) \qquad \mathrm{for }\; x\in \Omega, $$
$$u(t, 0) = 0 = u(t, 1) \qquad \mathrm{for }\; t \in \mathbb{T} .$$


## Setup / Input Data Generation

To setup the transport equation experiment please execute the two generation steps for the input data for training and evaluation
- input_data/generate_input_data.py
- input_data/test_data/generate_test_data.py

## Example execution

To train the neural network execute 

```
python3 certified_pinn.py -t user -u heat_equation/heat_equation.py train -i heat_equation/input_data/initial_data.csv
```

Then you can extract the key parameters via 
```
python3 certified_pinn.py -t user -u heat_equation/heat_equation.py extract -i heat_equation/input_data/initial_data.csv
```

and execute evaluation runs 
```
python3 certified_pinn.py -t user -u heat_equation/heat_equation.py run -i test_data/test_data_t_0.30.csv -ae domain
```

After executing all runs, you can aggregate the results to a csv 

```
python3 heat_equation/output_data/run_tanh_values/join_error_json.py
```
and plot the results
```
python3 heat_equation/output_data/run_tanh_values/plot_errors.py
```