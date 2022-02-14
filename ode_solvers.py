from xml.dom.expatbuilder import InternalSubsetExtractor
from ode import f, g, predator_prey
import numpy as np
import sys
import plots as p
import matplotlib.pyplot as plt
import time


def euler_step(X, t, h, f, params):
    '''
    Function that carries out one step of the Euler method.

    ARGS:   X = the value of the ODE's solution at time t.
            t = time at which to evaluate the gradient.
            h = the timestep to be carried out.
            f = the ODE that is being solved.
    
    RETURNS:    Xnew = the ODE's solution evaluated at t+h.
    
    EXAMPLE:    Xnew = euler_step(X=1, t=0, h=0.1, f)
                where f(X,t) = dX/dt
    '''
    # Work out the gradient and apply the Euler method step.
    dxdt = f(X,t, params)
    Xnew = X + h*np.array(dxdt)
    
    return Xnew

def RK4_step(X, t, h, f, params):
    '''
    Function that carries out one step of the RK4 numerical method.

    ARGS:   X = the value of the ODE's solution at time t.
            t = time at which to evaluate the gradient.
            h = the timestep to be carried out.
            f = the ODE that is being solved.
    
    RETURNS:    Xnew = the ODE's solution evaluated at t+h.
    
    EXAMPLE:    Xnew = RK4_step(X=1, t=0, h=0.1, f)
                where f(X,t) = dX/dt
    '''
    # Work out ks
    k1 = np.array( f( X , t , params) )
    k2 = np.array( f( X+h*(k1/2) , t+(h/2) , params) )
    k3 = np.array( f( X+h*(k2/2) , t+(h/2) , params ) )
    k4 = np.array( f( X+h*k3 , t+h , params ) )

    # Work out next X
    Xnew = X + (1/6) * h * (k1 + (2*k2) + (2*k3) + k4)
    
    return Xnew

def midpoint_step(X, t, h, f, params):
    '''
    Function that carries out one step of the midpoint numerical method.
    https://en.wikipedia.org/wiki/Midpoint_method

    ARGS:   X = the value of the ODE's solution at time t.
            t = time at which to evaluate the gradient.
            h = the timestep to be carried out.
            f = the ODE that is being solved.
    
    RETURNS:    Xnew = the ODE's solution evaluated at t+h.
    
    EXAMPLE:    Xnew = midpoint_step(X=1, t=0, h=0.1, f)
                where f(X,t) = dX/dt
    '''
    # calculate Xnew using formula
    Xnew = X + h*np.array(f(X+(h/2)*np.array(f(X,t)), t+(h/2), params))
    
    return Xnew

def heun3_step(X, t, h, f, params):
    '''
    Function that carries out one step of the Heun 3rd order method.

    ARGS:   X = the value of the ODE's solution at time t.
            t = time at which to evaluate the gradient.
            h = the timestep to be carried out.
            f = the ODE that is being solved.
    
    RETURNS:    Xnew = the ODE's solution evaluated at t+h.
    
    EXAMPLE:    Xnew = heun3_step(X=1, t=0, h=0.1, f)
                where f(X,t) = dX/dt
    '''
    # calculate Xnew using formula
    k1 = h*np.array(f(X, t, params))
    k2 = h*np.array(f(X+(k1/3), t+(h/3), params))
    k3 = h*np.array(f(X+(2*(k2/3)), t+(2*(h/3)), params))

    Xnew = X + k1/4 + 3*k3/4
    
    return Xnew


def solve_to(t0, t1, X0, h_max, f, method, params):
    '''
    Function that evaluates the solution to the ODE, X1, at time t1 given X0 and t0.
    i.e. the function carries out one iteration of the chosen numerical method.
    If the difference between t1 and t0 > h_max, multiple steps are carried out.

    ARGS:   t0 = the intial time.
            t1 = the time the ODE will be evaulated at.
            X0 = the value of X at t0.
            h_max = the maximum step size to be used to solve the ODE.
            f = the ODE to be solved
            method = the numerical method to use. (euler_step or RK4_step).
    
    RETURNS:    X1 = the solution to the ODE evaluated at t1.
    
    EXAMPLE:    X1 = solve_to(t0=0, t1=0.1, X0=1, h_max=0.1, f, euler_step)
                where f(X,t) = dX/dt
    '''
    # Find the time for the method to step over
    h = t1 - t0

    # Verify h is not greater than h_max
    if h > h_max:
        # Repeat until t1 is reached
        while np.round(t0,15) < np.round(t1, 15):
            # If applying method with step h_max will make t > t1, use different h,
            # such that the ODE is evaluated at t1 exactly
            if (t0 < t1) and (t0 + h_max > t1):
                X1 = method(X0, t0, t1-t0, f, params)
                t0 = t1

            # Apply method with step of h_max
            else:
                X1 = method(X0, t0, h_max, f, params)
                X0 = X1
                t0 += h_max

    # If h < h_max, compute next solution in one step
    else:
        X1 = method(X0, t0, h, f, params)

    return X1


def solve_ode(method, f, t, X0, h_max = 0.1, **params):
    '''
    Function that generates a series of numerical estimates to the ODE provided

    ARGS:   method = (string) the numerical method to use. ('euler'/'rk4'/'midpoint'/'heun3')
            f = the function containing the ODE to be solved.
            t = the timesteps to evaulate the ODE at.
            X0 = the initial coniditions of the ODE.
            h_max = the maximum step size to be used to solve the ODE.
    
    RETURNS:    X = list of solutions for every time in t.
    
    EXAMPLE:    X = solve_ode(euler_step, f, t=np.linspace(0,1,11), X0=1, h_max=0.1)
                where f(X,t) = dX/dt
    '''

    method_dict = { 'euler': euler_step,
                    'rk4': RK4_step,
                    'midpoint': midpoint_step,
                    'heun3': heun3_step}

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
        X[i+1] = solve_to(t0, t1, X[i], h_max, f, method_dict[method], params)

    return X


def evaluate_methods(methods, f, desired_tol, t0, t1, X0, X_true, **params):
    '''
    Function that takes multiple numericcal methods and assessed their speed and
    performance for a desired error tolerance. Produces a plot to show the required
    step size to reach the desired tolerance.

    ARGS:   methods = list of strings that give the methods to be assessed.
            f = the ODE to test the methods with.
            desirerd_tol = the desired tolerance to assess the methods at.
            t0 = the initial time to assess the ODE solution from.
            t1 = the end time for the ODE solution, where the error is calculated.
            X0 = the initial conditions for the ODE.
            X_true = the true value of X at t1.

    EXAMPLE: evaluate_methods(['euler', 'rk4'], f, desired_tol = 10**-4, 0, 1, 1, np.e)
    '''
    h_line = np.array([desired_tol]*200)
    # Plot the error vs h graph but do not show so more lines can be added.
    method_errors, hs = p.plot_error(methods, f, t0, t1, X0, X_true, show_plot=False)

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
            solve_ode(method, f, t, X0, h_max =hs[intersection] )
            end_time = time.time()
            print('Time taken to solve to desired tol: '+ str(end_time-start_time) + 's')
            # Plot the lines on the error vs h graph
            plt.axvline(hs[intersection], c='r', linestyle='--')

        i+=1
    
    # plot the desired tolerance line. 
    plt.axhline(h_line[0], linestyle = '--', c='k', label='Desired Tol')
    plt.legend(), plt.show()

    return 0



def main():

    t = np.linspace(0,6,61)
    X0 = np.array([0,1])

    X = solve_ode('heun3', g, t, X0)

    x0_true = np.sin(t)
    x1_true = np.cos(t)
    X_true = np.array([x0_true, x1_true]).transpose()

    p.plot_solution(t, X, 't', 'x0 and x1', 'Solution in Time', X_true)
    p.plot_solution(X[:,0], X[:,1], 'x0', 'x1', 'x1 against x0')

    t = np.linspace(0,1,101)
    X0= np.array([1])
    X = solve_ode('rk4', f, t, X0)
    X_true = np.e**(t)
    p.plot_solution(t, X, 't', 'x', 'Solution in Time RK4', X_true)

    X = solve_ode('euler', f, t, X0)
    X_true = np.e**(t)
    p.plot_solution(t, X, 't', 'x', 'Solution in Time E', X_true)

    t = np.linspace(0, 100, 1001)
    X = solve_ode('rk4', predator_prey, t, np.array([1,1]), a=1, b=0.3, d=0.1)
    p.plot_solution(t, X, 't', 'x and y', 'Predator-Prey Solution')

    #p.plot_error(['heun3', 'euler', 'rk4', 'midpoint'], g, 0, 1, np.array([0,1]), np.array([np.sin(1), np.cos(1)]))

    #evaluate_methods(['euler', 'rk4', 'heun3'], g, 10**-5, 0, 1, np.array([0,1]), np.array([np.sin(1), np.cos(1)]))

    return 0


if __name__ == '__main__':
    main()
