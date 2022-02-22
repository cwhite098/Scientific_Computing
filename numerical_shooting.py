from ode_solvers import solve_ode
import numpy as np
from scipy.optimize import fsolve
from ode import predator_prey, f, g, hopf
import plots as p



def root_finding_problem(X0, *data):
    '''
    Function that is a root finding problem, used in the numerical shooting method
    to find a limit cycle of an ODE. This function is to be passed to fsolve or another
    root finding method.

    Parameters
    ----------
    X0 : list
        The the initial guesses for the initial conditions
        of the limit cycle
    *data : tuple
        This contains: 
            f : function
                The function containing the ODE to be solved.
            phase_condition : function
                The phase condition for the limit cycle.
            **params:
                Any params needed to solve the ODE in f.

    Returns
    -------
    output : list
        The current predicted value of the initial conditions and
        period of the limit cycle.
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

def numerical_shooting(X0, T_guess, f, phase_condition, **params):
    '''
    Function that finds the period and initial conditions for a limit cycle of an ODE
    by constructing a root finding problem and solving it.

    Parameters
    ----------
    X0 : list
        The initial guess for the starting point of the limit cycle.
    T_guess : float
        The initial guess for the period of the limit cycle.
    f : function
        The function containing the ODE the limit cycle will be found for.
    phase_condition: function
        The function that acts as the final equation in the
        root finding problem.
    **params: 
        Any params required to solve the ODE.

    Returns
    -------
    X0 : list
        List containing the initial conditions for the limit cycle.
    T : float
        The period of the limit cycle.

    Example
    -------
    X0, T = numerical_shooting([1.5,1.5], 10, predator_prey, phase_condition, a=1, b=0.2, d=0.1)
    '''

    # Check for correct inputs
    if not callable(f):
        raise TypeError('f must be a function!')
    if not callable(phase_condition):
        raise TypeError('phase_condition must be a function!')
    try:
        X = f(X0, 0, params)
    except IndexError:
        raise ValueError('Dimensions of Initial Conditions do not match ODE!')


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


def pc_predator_prey(X0, **params):
    # returns dx/dt at t=0
    return predator_prey(X0, 0, params)[0]

def pc_g(X0, **params):
    # returns fixes y=1 at t=0
    return X0[1] -1

def pc_hopf(X0, **params):
    # returns du1dt at t=0 (to be set =0)
    return hopf(X0, 0, params)[0]


def main():
    a=1
    X0, T = numerical_shooting([1.3, 1.3], 23, predator_prey, pc_predator_prey, a=1, b=0.2, d=0.1)
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

    sigma = -1
    beta = 1
    t = np.linspace(0,1,101)
    X0, T = numerical_shooting([1,1], 5, hopf, pc_hopf, beta=beta, sigma=sigma)
    print(X0)
    print(T)
    t = np.linspace(0,T,1000)
    X = solve_ode('rk4', hopf, t, X0, beta=beta, sigma=sigma, h_max=0.001)
    p.plot_solution(t, X, 't', 'u1 and u2', 'Hopf Limit Cycle')

    return 0


if __name__ == '__main__':
    main()