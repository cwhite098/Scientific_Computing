from ode import f, g
import numpy as np
import sys
from plots import *


def euler_step(X, t, h, f):
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
    dxdt = f(X,t)
    Xnew = X + h*np.array(dxdt)
    
    return Xnew

def RK4_step(X, t, h, f):
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
    k1 = np.array(f(X,t))
    k2 = np.array(f(X+h*(k1/2), t+(h/2)))
    k3 = np.array(f(X+h*(k2/2), t+(h/2)))
    k4 = np.array(f(X+h*k3, t+h))

    # Work out next X
    Xnew = X + (1/6)*h*(k1 + (2*k2) + (2*k3) + k4)
    
    return Xnew

def midpoint_step(X, t, h, f):
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
    Xnew = X + h*np.array(f(X+(h/2)*np.array(f(X,t)), t+(h/2)))
    
    return Xnew


def solve_to(t0, t1, X0, h_max, f, method):
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
        while round(t0,15) < round(t1, 15):
            # If applying method with step h_max will make t > t1, use different h,
            # such that the ODE is evaluated at t1 exactly
            if (t0 < t1) and (t0 + h_max > t1):
                X1 = method(X0, t0, t1-t0, f)
                t0 = t1

            # Apply method with step of h_max
            else:
                X1 = method(X0, t0, h_max, f)
                X0 = X1
                t0 += h_max

    # If h < h_max, compute next solution in one step
    else:
        X1 = method(X0, t0, h, f)

    return X1


def solve_ode(method, f, t, X0, h_max = 0.1):
    '''
    Function that generates a series of numerical estimates to the ODE provided

    ARGS:   method = the numerical method to use. (euler_step or RK4_step).
            f = the function containing the ODE to be solved.
            t = the timesteps to evaulate the ODE at.
            X0 = the initial coniditions of the ODE.
            h_max = the maximum step size to be used to solve the ODE.
    
    RETURNS:    X = list of solutions for every time in t.
    
    EXAMPLE:    X = solve_ode(euler_step, f, t=np.linspace(0,1,11), X0=1, h_max=0.1)
                where f(X,t) = dX/dt
    '''

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
        X[i+1] = solve_to(t0, t1, X[i], h_max, f, method)

    return X



def main():

    t = np.linspace(0,6,61)
    X0 = np.array([0,1])

    X = solve_ode(midpoint_step, g, t, X0)

    x0_true = np.sin(t)
    x1_true = np.cos(t)
    X_true = np.array([x0_true, x1_true]).transpose()

    plot_solution(t, X, 't', 'x0 and x1', 'Solution in Time', X_true)
    plot_solution(X[:,0], X[:,1], 'x0', 'x1', 'x1 against x0')



    t = np.linspace(0,1,11)
    X0= np.array([1])
    X = solve_ode(midpoint_step, f, t, X0)
    X_true = np.e**t

    plot_solution(t, X, 't', 'x', 'Solution in Time', X_true)

    plot_error([euler_step, RK4_step, midpoint_step], f, 0, 1, np.array([1]), np.e)

    return 0


if __name__ == '__main__':
    main()
