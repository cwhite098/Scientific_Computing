# Scientific Computing - Numerical Toolbox for ODEs and PDEs

This package contains various methods for solving ODEs, PDEs as well as numerical shooting and numerical continuation.



## Implemented Methods
### Solving ODEs
For solving ODEs, the following methods are implemented:
 - The Euler Method,
 - The Midpoint Method,
 - The Heun 3rd Order Method,
 - The Runge-Kutte 4th Order Method.

These methods can be applied to 1st order ODEs, systems of 1st order ODEs and higher order ODEs that have been reduced to a system of 1st order ODEs.


### Numerical Shooting
Numerical shooting is the process of finding periodic orbits in the oscillatory solutions of ODEs or ODE systems.

This is carried out by defining a phase condition and constructing a root finding problem, where the roots are the initial conditions and period of the orbit.


### Numerical Continuation
Numerical continuation is the process of finding equilibria and periodic orbits of an ODE (or a system of ODEs) while varying a parameter.

Methods implemented include:
 - Natural Parameter Continuation,
 - Pseudo-Arclength Continuation.

These methods can be applied to 1st order ODEs, systems of 1st order ODEs, higher order ODEs that have been reduced to a system of 1st order ODEs as well as equations where the roots are found instead of equilibria or periodic orbits.


### Solving PDEs
For solving PDEs, the following methods are implemented:
 - Forward Euler Method,
 - Backward Euler Method,
 - Crank-Nicholson Method.

These methods can be applied to PDEs of the following types:
 - 2nd Order Diffusive PDEs with
    - One spatial dimension,
    - Homogeneous Dirichlet Boundary Conditions,
    - Non-homogeneous Dirichlet Boundary Conditions,
    - Homogeneous Neumann Boundary Conditions,
    - Non-homogeneous Neumann Boundary Conditions,
    - Periodic Boundary Conditions,
    - Robin Boundary Conditions,
    - Non-homogeneous and homogeneous RHS functions,
    - Linear and non-linear RHS functions,
    - Spatially varying diffusion coefficients.


### Plotting
Also included are functions that can be used to easily visualise the outputs of the numerical methods implemented.
Examples of their usage can be found in the example notebooks, detailed below.

## Requirements
Some existing Python packages are required to use this package. These include:
 - NumPy,
 - SciPy,
 - Matplotlib.

A requirements.txt [here](requirements.txt), is also provided for easy install of the required packages. This can be carried out using the following command:

```bash

$ pip install -r requirements.txt

```

## Examples
Examples for the use of all of the above methods can be found as Python Notebooks in the examples folder. This includes,
 - Examples for solving ODEs [here](examples/solving_odes.ipynb),
 - Examples for numerical shooting [here](examples/numerical_shooting.ipynb),
 - Examples for numerical continuation [here](examples/numerical_continuation.ipynb),
 - Examples for solving PDEs [here](examples/solving_pdes.ipynb).

These examples can be viewed straight in this repo, or downloaded where they can be run and tinkered with.
Examples can also be found by running the corresponding .py files as a script.


## Testing
Tests for the code in this package can be found in the tests.py file [here](tests.py). They are created using the built-in unittest class.

## Author
Christopher White

gd19031@bristol.ac.uk

## Citation
```
@software{Chris_White_SciComp,
  author = {White, Christopher},
  month = {3},
  title = {{Scientific Computing - Numerical Toolbox for ODEs and PDEs}},
  url = {https://github.com/cwhite098/Scientific_Computing},
  version = {1},
  year = {2022}
}
```
