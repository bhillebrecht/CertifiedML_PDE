# Klein Gordon Equation with special norm on H2

## Problem Description
We consider the Klein Gordon equation on the domain $\mathbb{T} \times \Omega = [0, 0.2] \times (0,1)$
$$\partial_{tt} u = \partial_{xx} u - \frac{1}{4} u  $$
$$u(0,x) = \cos(2 \pi x) ,$$
$$\partial_t u(0,x) = \tfrac{1}{2} \cos( 4 \pi x ) ,$$
$$\partial_x u (t, x\in\{0,1\}) =\;0 .$$ 
and the it is considered on the function space $H^2(\Omega)\times L^2(\Omega)$ with the norm 
$$\| (u, v) \| = \left( \| \nabla u\|^2_{L^2(\Omega)} + 0.25 \|u\|^2_{L^2(\Omega)} + \| v\|^2_{L^2(\Omega)}\right)^{1/2}$$
wherein we consider $v \coloneqq \partial_t u$.

## Setup / Input Data Generation

To setup the transport equation experiment please execute the two generation steps for the input data for training and evaluation
- input_data/generate_input_data.py
- input_data/reference_solution/generate_input_data_simple.py

## Example execution

To train the neural network execute 

```
python3 certified_pinn.py -t user -u klein_gordon_in_H2L2/klein_gordon.py train -i klein_gordon_in_H2L2/input_data/taylor_input.csv
```

Then you can extract the key parameters via 
```
python3 certified_pinn.py -t user -u klein_gordon_in_H2L2/klein_gordon.py extract -i klein_gordon_in_H2L2/input_data/kleingordon_input_simple.csv
```

and execute evaluation runs 
```
python3 certified_pinn.py -t user -u klein_gordon_in_H2L2/klein_gordon.py run -i eval_data/kleingordon_ref_t0.020_simple.csv -ae domain
```

After executing all runs, you can aggregate the results to a csv 

```
python3 klein_gordon_in_H2L2/output_data/run_tanh_values/join_error_json.py
```
and plot the results
```
python3 klein_gordon_in_H2L2/output_data/run_tanh_values/plot_errors.py
```