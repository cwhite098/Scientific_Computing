from xml.dom.expatbuilder import InternalSubsetExtractor
from ode import f, g, predator_prey, f2, hopf3D, modified_hopf
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

    t = np.linspace(0,6,61)
    X0 = [0,1]
    X = solve_ode('heun3', g,t, X0)

    x0_true = np.sin(t)
    x1_true = np.cos(t)
    X_true = [x0_true, x1_true]

    p.plot_solution(t, X, 't', 'x0 and x1', 'Solution in Time', X_true)
    p.plot_solution(X[:,0], X[:,1], 'x0', 'x1', 'x1 against x0')

    t = np.linspace(0,1,101)
    X0= [1]
    X = solve_ode('rk4', f, t, X0, h_max=0.001)
    X_true = np.e**(t)
    p.plot_solution(t, X, 't', 'x', 'Solution in Time RK4', X_true)

    X = solve_ode('euler', f, t, X0)
    X_true = np.e**(t)
    p.plot_solution(t, X, 't', 'x', 'Solution in Time E', X_true)

    t = np.linspace(0, 20.289717493033194, 1001)
    X = solve_ode('rk4', predator_prey, t, [0.27015621, 0.27015621], a=1, b=0.2, d=0.1, h_max=0.001)
    p.plot_solution(t, X, 't', 'x and y', 'Predator-Prey Solution')

    t = np.linspace(0, 20, 1001)
    X = solve_ode('rk4', modified_hopf, t, [1,1], beta=2)
    p.plot_solution(t, X, 't', 'x and y', 'Modified Hopf Solution')

    X = solve_ode('rk4', hopf3D, t, [1,0,1], beta=1, sigma=-1, h_max=0.001)
    p.plot_solution(t, X, 't', 'u1, u2, u3', 'Hopf3D Solution')

   # p.plot_error(['heun3', 'euler'], g, 0, 1, np.array([0,1]), np.array([np.sin(1), np.cos(1)]))

    #p.plot_error(['rk4', 'euler'], f, 0, 1, np.array([1]), np.e)

    #p.plot_error(['euler', 'rk4'], f2, 0, 1, np.array([1]), np.e, a=1)

    #evaluate_methods(['euler', 'rk4'], f, 10**-5, 0, 1, [1], np.e, a=1)

    return 0


if __name__ == '__main__':
    main()
