from ode_solvers import solve_ode
import numpy as np
from scipy.optimize import fsolve
from ode import predator_prey



def numerical_shooting(X0):

    T = 20
    t = np.linspace(0,T,3)

    solution = solve_ode('rk4', predator_prey, t, X0, a=1, b=0.2, d=0.1, hmax=0.001)
    
    output = X0 - solution[-1,:]

    return output


def find_vars():

    X0=[0.1, 1]
    sol = fsolve(numerical_shooting, X0 )

    return sol




def main():

    sol = find_vars()
    print(sol)

    return 0


if __name__ == '__main__':
    main()