# Transport Equation (2 dim)

## Problem Description

The boundary and initial value problem under consideration is
$$\partial_t u(t,\mathbf{x}) =\; (\tfrac{1}{5}, \tfrac{1}{2}) \cdot \nabla u(t,\mathbf{x})\quad (t,\mathbf{x}) \in \mathbb{T} \times \Omega,$$
$$u(0,\mathbf{x}) =\;  \begin{cases}
        \tfrac{1}{2} - \| \mathbf{x} \|_1, \quad\quad\;\; &\mathrm{for }\; \| \mathbf{x} \|_1 \le \tfrac{1}{2}, \\
        0, \quad\quad &\mathrm{else}, 
        \end{cases} $$
$$ u(t, -2, x_2 ) = \; u(t, 2, x_2) \qquad\quad\;\, t\in\mathbb{T},\; x_2 \in [-2,2] , $$
$$ u(t, x_1, -2) = \; u(t, x_1, 2) \qquad\quad\;\,  t\in\mathbb{T},\; x_1 \in [-2,2] ,$$

for $\mathbb{T} = [0,8]$ and training domain $\tilde{\mathbb{T}} =[0,4]$. The periodic boundary conditions are enforced as hard constraints. 

## Setup / Input Data Generation

To setup the transport equation experiment please execute the two generation steps for the input data for training and evaluation
- input_data/generate_input_data.py
- input_data/timeseries_input_data/generate_test_data.py

## Example execution

To train the neural network execute 

```
python3 certified_pinn.py -t user -u transport_equation_2dim/transport_equation.py train -i transport_equation_2dim/input_data/hat_input.csv
```

Then you can extract the key parameters via 
```
python3 certified_pinn.py -t user -u transport_equation_2dim/transport_equation.py extract -i transport_equation_2dim/input_data/hat_input.csv
```

and execute evaluation runs 
```
python3 certified_pinn.py -t user -u transport_equation_2dim/transport_equation.py run -i timeseries_input_data/input_eval_1.60.csv -ae domain
```

After executing all runs, you can aggregate the results to a csv 

```
python3 transport_equation_2dim/output_data/run_tanh_values/join_error_json.py
```