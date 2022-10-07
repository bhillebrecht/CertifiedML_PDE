# Navier Stokes Equations (Taylor Flow)

## Problem Description
We consider the Navier Stokes equation in the velocity pressure formulation
$$
\partial_t \mathbf{u} + \varrho (\mathbf{u} \cdot \nabla) \mathbf{u} = - \nabla p + \frac{1}{Re} \nabla^2 \mathbf{u} \qquad \mathrm{in }\; \mathbb{T} \times \Omega, $$
$$\nabla \cdot \mathbf{u} = 0  \mathrm{in }\; \mathbb{T} \times \Omega,$$
for $Re = 1$, $\varrho = 1$ and the initial and boundary conditions corresponding to the Taylor flow
$$u(t, x_1, x_2) = - \mathrm{cos}(x_1) \sin (x_2) \mathrm{e}^{-2t}, $$
$$v(t, x_1, x_2) = \sin(x_1) \cos (x_2) \mathrm{e}^{-2t}, $$
$$p(t, x_1, x_2) = - \frac{1}{4}( \mathrm{cos}(2 x_1) + \cos (2 x_2) )\mathrm{e}^{-4t} .$$

## Setup / Input Data Generation

To setup the transport equation experiment please execute the two generation steps for the input data for training and evaluation
- input_data/generate_input_data.py
- input_data/eval_data/generate_test_data.py

## Example execution

To train the neural network execute 

```
python3 certified_pinn.py -t user -u nse/nse.py train -i nse/input_data/taylor_input.csv
```

Then you can extract the key parameters via 
```
python3 certified_pinn.py -t user -u nse/nse.py extract -i nse/input_data/taylor_input.csv
```

and execute evaluation runs 
```
python3 certified_pinn.py -t user -u nse/nse.py run -i eval_data/taylor_t0.15.csv -ae domain
```

After executing all runs, you can aggregate the results to a csv 

```
python3 nse/output_data/run_tanh_values/join_error_json.py
```
and plot the results
```
python3 nse/output_data/run_tanh_values/plot_errors.py
```