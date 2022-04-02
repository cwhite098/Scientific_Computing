from ode_solvers import solve_ode
import numpy as np
from scipy.optimize import fsolve
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

    # No extra parameters provided
    if len(data)==2:
        f, phase_condition = data
        # Solve the ODE up until t=T
        solution = solve_ode('rk4', f, t, X0, hmax=0.001)
        # Construct the output, X(0) - X(T) and the phase condition
        output = np.append(X0 - solution[-1,:], phase_condition(X0))

    # Extra parameters provided
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
    phase_condition : function
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
        
    If the root finding process does not converge, an empty array is
    returned.

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

    # Check for convergence
    if sol[:-1].all() == np.array(X0).all() and sol[-1] == T_guess:
        print('Root Finder Failed, returning empty array...')
        return []

    # Split back into ICs and T
    X0 = sol[:-1]
    T = sol[-1]

    return X0, T



def main():

    # ## EXAMPLES
    # 
    # This package contains methods for numerically finding limit cycles in oscillatory solutions of ODEs.

    # It is necessary to first define your 1st order (or system of 1st order) ODE(s). For the following example, a predator-prey model will be used:
    # 
    # $\frac{dx}{dt} = x(1-x) - \frac{axy}{d+x}$,
    # 
    # $\frac{dy}{dt} = by(1-\frac{y}{x})$,
    # 
    # with parameter values of $a=1$, $d=0.1$ and $b=0.2$. This system of ODEs can be encoded as a Python function as shown in the next cell.

    def predator_prey(X, t, params):
        '''
        Function that contains the predator-prey ODE system.

        Parameters
        ----------
        X : list
            The current state of the system (values of x and y).
        t : np.array
            Numpy array containing the times for the solution to be evaluated at.
        params : dict
            Dictionary containing the parameters required for solving the system.
        
        Returns
        -------
        X : list
            List containing the gradients computed for the given values of x, y and the 
            parameters.
        '''
        # Get parameters
        a = params['a']
        b = params['b']
        d = params['d']

        # Get system state
        x = X[0]
        y = X[1]

        # Calculate gradients
        dxdt = x*(1-x) - (a*x*y) / (d+x)
        dydt = b*y*(1-(y/x))

        X = [dxdt, dydt]
        return X

    # As well as defining the system of ODEs, it is also necessary to define a phase condition since 
    # otherwise there is not enough information for the numerical shooting method to produce an unique periodic orbit.
    # 
    # We will use the phase condition, $\frac{dx(t=0)}{dt} = 0$. This will fix the gradient of $x$ at time $t=0$ to $0$.

    def pc_predator_prey(X0, **params):
        '''
        Function containing the phase conidition that fixes dx/dt=0 for t=0.

        Parameters
        ----------
        X0 : list
            Initial conditions of the system [x,y].
        **params:
            Optional parameters for passing to the ODE system.
            In this case, a, b and d are required but these will be defined when the
            numerical shooing occurs.

        Returns
        -------
        dxdt_at_0 : float
            The value of the gradient in the x direction at t=0. 
            This will be set to 0 as the root finding part of the shooting algorithm converges.
        '''
        dxdt_at_0 = predator_prey(X0, 0, params)[0]
        return dxdt_at_0

    # We are now ready to carry out the numerical shooting. This can be done as shown below.

    X0, T = numerical_shooting([1.3, 1.3], 10, predator_prey, pc_predator_prey, a=1, b=0.2, d=0.1)
    print(X0)
    print(T)

    # The numerical shooting function returns X0 which contains the initial conditions for the periodic orbit and 
    # T which is the period of the periodic orbit. To visualise this solution, the plot_orbit function can be used.

    p.plot_orbit(X0, T, predator_prey, 'Periodic Orbit', a=1, b=0.2, d=0.1)

    # This method can be used to find periodic orbits in ODE systems with arbitrary dimensions.






if __name__ == '__main__':
    main()