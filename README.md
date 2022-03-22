## Scientific Computing - Numerical Methods for ODEs and PDEs

This package contains various methods for solving ODEs, PDEs as well as numerical shooting and numerical continuation.



### Implemented Methods
#### Solving ODEs
For solving ODEs, the following methods are implemented:
 - The Euler Method,
 - The Midpoint Method,
 - The Heun 3rd Order Method,
 - The Runge-Kutte 4th Order Method.

These methods can be applied to 1st order ODEs, systems of 1st order ODEs and higher order ODEs that have been reduced to a system of 1st order ODEs.


#### Numerical Shooting


#### Numerical Continuation
Numerical continuation is the process of finding equilibria and periodic orbits of an ODE (or a system of ODEs) while varying a parameter.

Methods implemented include:
 - Natural Parameter Continuation,
 - Pseudo-Arclength Continaution.

These methods can be applied to 1st order ODEs, systems of 1st order ODEs and higher order ODEs that have been reduced to a system of 1st order ODEs.


#### Solving PDEs
For solving PDEs, the following methods are implememted:
 - Forward Euler Method,
 - Backward Euler Method,
 - Crank-Nicholson Method.

These methods can be applied to PDEs of the following types:
 - 2nd Order Diffusive PDEs with
    - Homogeneous Dirichlet Boundary Conditions ($u(x,0) = u(x,L) = 0$),
    - One spatial dimension,
    - and more to come...

### Requirements
Some existing Python packages are required to use this package. These include:
 - NumPy,
 - SciPy,
 - Matplotlib.
A requirements.txt is also provided for easy install of the required packages. This can be carried out using the following command:

PUT COMMAND IN (ALSO MAKE REQUIREMENTS.TXT)!!!!



### Examples
Examples for the use of all of the above methods can be found as Python Notebooks in the examples folder.


### Author
Christopher White
gd19031@bristol.ac.uk
