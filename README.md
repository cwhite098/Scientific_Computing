# Scientific Computing - Numerical Methods for ODEs and PDEs

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
 - Pseudo-Arclength Continaution.

These methods can be applied to 1st order ODEs, systems of 1st order ODEs, higher order ODEs that have been reduced to a system of 1st order ODEs as well as equations where the roots are found instead of equilibria or periodic orbits.


### Solving PDEs
For solving PDEs, the following methods are implememted:
 - Forward Euler Method,
 - Backward Euler Method,
 - Crank-Nicholson Method.

These methods can be applied to PDEs of the following types:
 - 2nd Order Diffusive PDEs with
    - One spatial dimension,
    - Homogeneous Dirichlet Boundary Conditions,
    - Non-homogeneous Dirichlet Boundary Conditions,
    - Homogeneous Neuamnn Boundary Conditions,
    - Non-homogeneous Neumann Boundary Conditions,
    - Periodic Boundary Conditions,
    - Robin Boundary Conditions,
    - and more to come...


## Requirements
Some existing Python packages are required to use this package. These include:
 - NumPy,
 - SciPy,
 - Matplotlib.

A requirements.txt [here](requirements.txt), is also provided for easy install of the required packages. This can be carried out using the following command:

'''bash

$ pip install -r requirements.txt

'''



## Examples
Examples for the use of all of the above methods can be found as Python Notebooks in the examples folder. This includes,
 - Examples for solving ODEs [here](examples/solving_odes.ipynb),
 - Examples for numerical shooting [here](examples/numerical_shooting.ipynb),
 - Examples for numerical continuation [here](examples/numerical_continuation.ipynb),
 - more to come.

These examples can be viewed straight in this repo, or downloaded where they can be run and tinkered with.


## Testing
Tests for the code in this package can be found in the tests.py file [here](tests.py). They are created using the built-in unittest class.


## References

## Author
Christopher White

gd19031@bristol.ac.uk
