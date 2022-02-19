from ode_solvers import solve_ode
import numpy as np
from scipy.optimize import fsolve
from ode import predator_prey, f, g
import plots as p



def root_finding_problem(X0, *data):
    '''
    Function that is a root finding problem, used in the numerical shooting method
    to find a limit cycle of an ODE. This function is to be passed to fsolve or another
    root finding method.

    ARGS:   X0 = a list containing the initial guesses for the initial conditions
                 of the limit cycle
            *data = tuple containing (f, phase_condition, **params)
                    where:  f = the ODE to be solved
                            phase_condition  = the phase condition for the
                                               shooting process
                            **params = any params need to solve f.

    EXAMPLE: X0, T = numerical_shooting([1.5,1.5], 10, predator_prey, phase_condition, a=1, b=0.2, d=0.1)
    '''

    # Init Variables
    T = X0[-1]    # Period of limit cycle
    X0 = X0[:-1]    # ICs of ODE
    t = np.linspace(0,T,3)    # time to solve ODE over

    if len(data)==2:
        f, phase_condition = data
        # Solve the ODE up until t=T
        solution = solve_ode('rk4', f, t, X0, hmax=0.001)
        # Construct the output, X(0) - X(T) and the phase condition
        output = np.append(X0 - solution[-1,:], phase_condition(X0))

    else:
        f, phase_condition, params = data
        # Solve the ODE up until t=T
        solution = solve_ode('rk4', f, t, X0, **params, hmax=0.001)
        # Construct the output, X(0) - X(T) and the phase condition
        output = np.append(X0 - solution[-1,:], phase_condition(X0, **params))
    
    return output


def pc_predator_prey(X0, **params):
    # returns dx/dt at t=0
    return predator_prey(X0, 0, params)[0]

def pc_g(X0, **params):
    # returns fixes y=1 at t=0
    return X0[1] -1


def numerical_shooting(X0, T_guess, f, phase_condition, **params):
    '''
    Function that finds the period and initial conditions for a limit cycle of an ODE
    by constructing a root finding problem and solving it.

    ARGS:   X0 = the initial guess for the starting point of the limit cycle.
            T_guess = the initial guess for the period of the limit cycle.
            f = the ODE the limit cycle will be found for.
            phase_condition = the function that acts as the final equation in the
                              root finding problem
            **params = any params required to solve the ODE.

    EXAMPLE: X0, T = numerical_shooting([1.5,1.5], 10, predator_prey, phase_condition, a=1, b=0.2, d=0.1)
    '''
    # Add T to intial conditions
    X0.append(T_guess)

    # Pass params in data if they are needed for the ODE
    if params:
        data = (f, phase_condition, params)
    else:
        data = (f, phase_condition)

    # Solve for the limit cycle using fsolve
    sol = fsolve(root_finding_problem, X0, args=data)

    # Split back into ICs and T
    X0 = sol[:-1]
    T = sol[-1]

    return X0, T




def main():

    X0, T = numerical_shooting([1.3,1.3], 23, predator_prey, pc_predator_prey, a=1, b=0.2, d=0.1)
    print(X0)
    print(T)

    t = np.linspace(0,T,1000)
    X = solve_ode('rk4', predator_prey, t, X0, a=1, b=0.2, d=0.1, h_max=0.001)
    p.plot_solution(t, X, 't', 'x and y', 'Predator-Prey Limit Cycle')



    X0, T = numerical_shooting([0.5,0.5], 7, g, pc_g)
    print(X0)
    print(T)

    t = np.linspace(0,T,1000)
    X = solve_ode('rk4', g, t, X0, h_max=0.001)
    p.plot_solution(t, X, 't', 'x and y', 'G Limit Cycle')

    return 0


if __name__ == '__main__':
    main()