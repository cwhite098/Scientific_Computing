import numpy as np
import sys
import plots as p
import matplotlib.pyplot as plt
import time


def euler_step(X, t, h, f, **params):
    '''
    Function that carries out one step of the Euler method.

    Parameters
    ----------
    X : np.array
        The value of the ODE's solution at time t.
    t : float
        Time at which to evaluate the gradient.
    h : float
        The timestep to be carried out.
    f : function
        The function containing the ODE that is being solved.

    Returns
    -------    
    Xnew : np array 
        The ODE's solution evaluated at t+h.
    
    Example
    -------    
    Xnew = euler_step(X=1, t=0, h=0.1, f)
    '''
    # Work out the gradient and apply the Euler method step.
    dxdt = f(X,t, params)
    Xnew = X + h*np.array(dxdt)
    
    return Xnew

def RK4_step(X, t, h, f, **params):
    '''
    Function that carries out one step of the
    Runge-Kutte 4th order method.

    Parameters
    ----------
    X : np.array
        The value of the ODE's solution at time t.
    t : float
        Time at which to evaluate the gradient.
    h : float
        The timestep to be carried out.
    f : function
        The function containing the ODE that is being solved.

    Returns
    -------    
    Xnew : np array 
        The ODE's solution evaluated at t+h.
    
    Example
    ------- 
    Xnew = RK4_step(X=1, t=0, h=0.1, f)
    '''
    # Work out ks
    k1 = np.array( f( X , t , params) )
    k2 = np.array( f( X+h*(k1/2) , t+(h/2) , params) )
    k3 = np.array( f( X+h*(k2/2) , t+(h/2) , params ) )
    k4 = np.array( f( X+h*k3 , t+h , params ) )

    # Work out next X
    Xnew = X + (1/6) * h * (k1 + (2*k2) + (2*k3) + k4)
    
    return Xnew

def midpoint_step(X, t, h, f, **params):
    '''
    Function that carries out one step of the midpoint method.

    Parameters
    ----------
    X : np.array
        The value of the ODE's solution at time t.
    t : float
        Time at which to evaluate the gradient.
    h : float
        The timestep to be carried out.
    f : function
        The function containing the ODE that is being solved.

    Returns
    -------    
    Xnew : np array 
        The ODE's solution evaluated at t+h.
    
    Example
    -------     
    Xnew = midpoint_step(X=1, t=0, h=0.1, f)
    '''
    # calculate Xnew using formula
    Xnew = X + h*np.array(f(X+(h/2)*np.array(f(X,t, params)), t+(h/2), params))
    
    return Xnew

def heun3_step(X, t, h, f, **params):
    '''
    Function that carries out one step of the
    Heun 3rd order method.

    Parameters
    ----------
    X : np.array
        The value of the ODE's solution at time t.
    t : float
        Time at which to evaluate the gradient.
    h : float
        The timestep to be carried out.
    f : function
        The function containing the ODE that is being solved.

    Returns
    -------    
    Xnew : np array 
        The ODE's solution evaluated at t+h.
    
    Example
    -------     
    Xnew = heun3_step(X=1, t=0, h=0.1, f)
    '''
    # calculate Xnew using formula
    k1 = h*np.array(f(X, t, params))
    k2 = h*np.array(f(X+(k1/3), t+(h/3), params))
    k3 = h*np.array(f(X+(2*(k2/3)), t+(2*(h/3)), params))

    Xnew = X + k1/4 + 3*k3/4
    
    return Xnew


def solve_to(t0, t1, X0, f, method, **params):
    '''
    Function that evaluates the solution to the ODE, X1, at time t1 given X0 and t0.
    i.e. the function carries out one iteration of the chosen numerical method.
    If the difference between t1 and t0 > h_max, multiple steps are carried out.

    Parameters
    ----------
    t0: float
        The intial time to start the solution step.
    t1 : float
        The time the ODE will be evaulated at.
    X0 : np.array
        The value of X at t0.
    f : function
        The function containing the ODE to be solved.
    method : string 
        The numerical method to use. (euler_step, RK4_step, midpoint_step or heun3_step).
    **params:
        h_max : float
            The maximum step size to use. The default value is 0.1.
        Any other parameters that need to be passed to the ODE function.
    
    Returns
    -------   
    X1 : np.array 
        The solution to the ODE evaluated at t1.
    
    Example 
    -------   
    X1 = solve_to(t0=0, t1=0.1, X0=1, f, euler_step, h_max=0.001)
    '''
    try:
        h_max = params['h_max']
    except KeyError:
        h_max = 0.1

    # Find the time for the method to step over
    h = t1 - t0

    # Verify h is not greater than h_max
    if h > h_max:
        # Repeat until t1 is reached
        while np.round(t0,15) < np.round(t1, 15):
            # If applying method with step h_max will make t > t1, use different h,
            # such that the ODE is evaluated at t1 exactly
            if (t0 < t1) and (t0 + h_max > t1):
                X1 = method(X0, t0, t1-t0, f, **params)
                t0 = t1

            # Apply method with step of h_max
            else:
                X1 = method(X0, t0, h_max, f, **params)
                X0 = X1
                t0 += h_max

    # If h < h_max, compute next solution in one step
    else:
        X1 = method(X0, t0, h, f, **params)

    return X1


def solve_ode(method, f, t, X0, **params):
    '''
    Function that generates a series of numerical
    estimates to the solution of the ODE provided.

    Parameters
    ----------
    method : string 
        The numerical method to use. ('euler'/'rk4'/'midpoint'/'heun3')
    f : function
        The function containing the ODE to be solved.
    t : np.array
        The timesteps to evaulate the ODE at.
    X0 : np.array
        The initial coniditions of the ODE.
    **params:
        h_max : float 
            The maximum step size to use in the solution. The default value is 0.1.
        Any parameters that need to be passed to the function containing the ODE.
    
    Returns
    -------
    X : np.array
        List of X values, evaluated at every time in t.
    
    Example
    -------
    X = solve_ode('euler', f, t=np.linspace(0,1,11), X0=1, h_max=0.1)
    '''
    method_dict = { 'euler': euler_step,
                    'rk4': RK4_step,
                    'midpoint': midpoint_step,
                    'heun3': heun3_step}

    # Check for correct inputs
    if not callable(f):
        raise TypeError('f must be a function!')
    try:
        X = f(X0, t, params)
    except IndexError:
        raise ValueError('Dimensions of Initial Conditions do not match ODE!')
    try:
        method = method_dict[method]
    except KeyError:
        raise KeyError('Specified incorrect solver!')
    
    # Make the initial conditions and n array
    X0 = np.array(X0)

    # Initialise the solution vector and add initial condition
    if len(X0) > 1:
        # If the ODE is 2nd order or more / a system of ODEs
        X = np.zeros((len(t), X0.shape[0]))
        X[0,:] = X0
    else:
        # If the ODE is 1st order
        X = np.zeros(len(t))
        X[0] = X0

    # Loop through the time vector and update the solution vector
    for i in range(len(t)-1):
        t0 = t[i]
        t1 = t[i+1]
        X[i+1] = solve_to(t0, t1, X[i], f, method, **params)

    return X


def evaluate_methods(methods, f, desired_tol, t0, t1, X0, X_true, **params):
    '''
    Function that takes multiple numerical methods and assesses their speed and
    performance for a desired error tolerance. Produces a plot to show the required
    step size to reach the desired tolerance.

    Parameters
    ----------
    methods : list 
        List of strings that give the methods to be assessed.
    f : function 
        The function containing the ODE to test the methods with.
    desirerd_tol : float
        The desired tolerance to assess the methods at.
    t0 : float 
        The initial time to assess the ODE solution from.
    t1 : float 
        The end time for the ODE solution, where the error is calculated.
    X0 : list
        The initial conditions for the ODE.
    X_true : list
        The true value of X at t1.
    **params:
        Any parameters that are required to solve the ODE.

    Example
    -------
    evaluate_methods(['euler', 'rk4'], f, desired_tol = 10**-4, 0, 1, 1, np.e)
    '''
    h_line = np.array([desired_tol]*200)
    # Plot the error vs h graph but do not show so more lines can be added.
    method_errors, hs = p.plot_error(methods, f, t0, t1, X0, X_true, show_plot=False, **params)

    i = 0
    # Loop through the returned errors for each method
    for errs in method_errors:
        method = methods[i]
        errs = np.array(errs)
        # Find the intersection between the error lines and the desired tolerance
        intersection = np.argwhere(np.diff(np.sign(h_line-errs))).flatten()

        print('\nMethod:',method)

        # If method can reach the tolerance print details - time and required h
        if not intersection.size > 0:
            print('Method cannot reach desired tol')
        else:
            print('h to meet desired tol:', hs[intersection])
            # do a quick solve using the hs[intersection] and time it, print the times
            start_time = time.time()
            t=np.linspace(t0,t1,10)
            solve_ode(method, f, t, X0, h_max = hs[intersection], **params)
            end_time = time.time()
            print('Time taken to solve to desired tol: '+ str(end_time-start_time) + 's')
            # Plot the lines on the error vs h graph
            plt.axvline(hs[intersection], c='r', linestyle='--')

        i+=1
    
    # plot the desired tolerance line. 
    plt.axhline(h_line[0], linestyle = '--', c='k', label='Desired Tol')
    plt.legend(), plt.show()




def main():

    ## EXAMPLES

    # ### Solving First Order ODEs
    # 
    # To solve an ODE, firstly it must be defined as a Python function. For example, the simple, 1st order ODE:
    # 
    # $\frac{dx}{dt} = x$
    # 
    # can be written as a Python function as follows:

    def f(x, t, params):
        '''
        Function containing ODE dx/dt=x.

        Parameters
        ----------
        x : list
            The values for the system (x). For this ODE the list will only have a single entry.
        t : np.array
            Numpy array containing the times the solution must be evaluated at.
        params : dict
            Any parameters needed to solve this ODE. For this example there are none.

        Returns
        -------
        x : float
            The value of dx/dt for the given x.
        '''
        return x

    # The function, f, returns the right hand side of the ODE.
    # 
    # Next, the time over which the solution is to be produced and an initial condition must be defined as follows:

    t = np.linspace(0,1,10) # Solution between t=0 and t=1 with 101 steps.
    X0 = [1] # The initial condition for the solution x(t=0) = 1.

    # We are now ready to solve the ODE. There are four methods that could be used to produce the solution. They are the Euler method, the Runge-Kutte 4th order method, the Heun 3rd order method and the midpoint method. For this initial example, the Euler method will be used.

    X = solve_ode('euler', f, t, X0, h_max=0.1)

    # In order to verify this solution, it can be compared against the analytical solution,
    # 
    # $x(t) = e^{t}$,
    # 
    # by plotting both the numerical and analytical solutions against time using the plot_solution function.

    X_analytical = np.e**t

    p.plot_solution(t, X, 't', 'x(t)', 'First Order ODE Solution', X_analytical)

    # It is also possible to apply this method to an ODE that contains a parameter, for example the ODE,
    # 
    # $\frac{dx}{dt} = ax$,
    # 
    # can be solved as below, with the parameter $a$ being passed as an optional argument.

    # Define the function that contains the ODE to be solved.
    # When defining an ODE that requires some parameter, the parameters are passed as a dictionary and
    # need to be extracted from said dictionary in the function, as below.
    def f2(x,t, params):
        '''
        Function containing ODE dx/dt=ax.

        Parameters
        ----------
        x : list
            The values for the system (x). For this ODE the list will only have a single entry.
        t : np.array
            Numpy array containing the times the solution must be evaluated at.
        params : dict
            Any parameters needed to solve this ODE. For this example only one is required, a.
            e.g. {'a' : 2}

        Returns
        -------
        x : float
            The value of dx/dt for the given x.
        '''
        a = params['a']
        return x*a

    t = np.linspace(0,1,10)
    X0 = [1]

    X = solve_ode('midpoint', f2, t, X0, h_max=0.1, a=2) # Pass the value of parameter a

    X_analytical = np.e**(2*t)

    p.plot_solution(t, X, 't', 'x(t)', 'First Order ODE Solution', X_analytical)

    # ### Solving Higher Order or Systems of ODEs
    # 
    # To solve higher order ODEs, it is necessary to reduce the equation into a system of first order ODEs.
    # 
    # For example, the second order ODE,
    # 
    # $\frac{d^{2}x}{dt^{2}} = -x$,
    # 
    # can be rewritten as a system of first order ODEs,
    # 
    # $\frac{dx}{dt} = y$,
    # 
    # $\frac{dy}{dt} = -x$.
    # 
    # This system of ODEs can be encoded as a Python function that can be passed to the solve_ode function.
    # 
    # 

    def g(X, t, params):
        '''
        Function containing 2nd order ODE d2x/dt2 = -x, reduced to a system of first order ODEs.

        Parameters
        ----------
        x : list
            The values of the system (x and y). For this ODE the list will have two entries
        t : np.array
            Numpy array containing the times the solution must be evaluated at.
        params : dict
            Any parameters needed to solve this ODE. For this example there are none.

        Returns
        -------
        x : list
            The value of dx/dt and dy/dt for given values of X and Y
        '''
        x = X[0]
        y = X[1]
        dxdt = y
        dydt = -x
        X = [dxdt, dydt]

        return X

    t = np.linspace(0,10,30)
    X0 = [0,1]

    X = solve_ode('heun3', g, t, X0) # No need to specify h_max, the default is h_max=0.1

    # Construct the analytical solution
    x1_analytical = np.sin(t)
    x2_analytical = np.cos(t)
    X_analytical = [x1_analytical, x2_analytical]

    p.plot_solution(t, X, 't', 'x(t)', 'First Order ODE Solution', X_analytical)

    # It is also possible to use the plot_solution function to plot one state variable against another, for example,

    p.plot_solution(X[:,0], X[:,1], 'x', 'y', 'y against x')

    # Solve ODE will work for systems of ODEs of arbitrary size and with any number of parameters.

    # ### Evaluating The Methods
    # 
    # This package contains four different methods for numerically solving ODEs. It also contains functions to demonstrate 
    # the differences between the methods in terms of errors and runtime.
    # 
    # There are two functions for this application.
    # 
    # Firstly, plot_error will plot the errors produced by the methods against a varying step size, h.

    # Show the diference in performance between the Heun 3rd order method and the Euler method.
    # The previously defined 2nd order ODE is used.
    # This function may take some time to run (approx 2 minutes)...
    method_errors, hs = p.plot_error(['heun3', 'euler'], g, 0, 1, np.array([0,1]), np.array([np.sin(1), np.cos(1)]))

    # It can be seen in the above plot that the Heun method is indeed 3rd order and the Euler method is 1st order. 
    # This can be calculated by inspecting the gradients of the lines plotted.
    # 
    # For a more comprhensive evaluation of the methods, for a specific tolerance, the evaluate_methods function can be used.

    # Evaluated the Euler and RK4 methods to a tolerance of 10^{-5}.
    # This function may also take some time (approx 3 minutes)...
    evaluate_methods(['euler', 'rk4', 'heun3'], f, 10**-5, 0, 1, [1], np.e)

    # It can be seen, as expected, the higher order methods reach the specified tolerance where the Euler method does not. 
    # RK4 can reach the tolerance with a much greater step size and less time required than the Heun 3rd order method.



if __name__ == '__main__':
    main()
