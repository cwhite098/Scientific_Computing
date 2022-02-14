import matplotlib.pyplot as plt
import numpy as np
import ode_solvers as ode
from ode import *


def plot_solution(t, X, xlabel='t', ylabel='x', title='Solution', X_true=None):
    '''
    Function that plots the numerical solution to an ODE as well as the true solution
    if required.

    ARGS:   t = the time over the ODE has been solved.
            X = the numerical solution at each t_i in t.
            xlabel = the label for the x-axis (string)
            ylabel = the label for the y-axis (string)
            title = the title for the plot (string)
            X_true = the true solution at each t_i in t.
    
    EXAMPLE:    plot_solution(t=linspace(0,1,11), X, X_true)
    '''

    # Check to see how many variables are to be plotted
    if len(X.shape) > 1:
        number_of_vars = X.shape[1]
        # If true solution is provided plot both X and X_true
        if X_true is not None:
            for i in range(number_of_vars):
                plt.plot(t, X_true[:,i], label='True Solution, x'+str(i))
                plt.plot(t, X[:,i], label='Approx Solution, x'+str(i))
            
        # No X_true provided, just plot X
        else:
            for i in range(number_of_vars):
                plt.plot(t, X[:,i], label='Approx Solution, x'+str(i))

    else:
        number_of_vars = 1
        # If true solution is provided plot both X and X_true
        if X_true is not None:
            plt.plot(t, X_true[:], label='True Solution')
            plt.plot(t, X[:], label='Approx Solution')

        # No X_true provided, just plot X
        else:
            plt.plot(t, X[:], label='Approx Solution')
    
    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel), plt.legend()
    plt.show()

    return 0


def plot_error(methods, f, t0, t1, X0, X1_true,  **params):
    '''
    Function that works out the global error for multiple numerical methods used to solve ODEs
    and plots the error against the step size on a loglog graph.

    ARGS:   method = list of methods to plot on the loglog plot.
            f = the function of the ODE that is to be evaluated.
            t0 = the initial time for the solution.
            t1 = the final time of the solution, this is where the error is calculated.
            X0 = (list) the initial conditions of the ODE.
            X1_true = the true value of the ODE's solution at t=t1.
            **params:   show_plot = bool that controls whether the plot is shown or not.
                        any parameters necessary for the ODE being solved

    RETURNS:    method_errors = the errors at t1 of all the methods for differnt hs

    EXAMPLE:    plot_error(['rk4', 'euler'], f, t0=0, t1=1, X0=1, X1_true=np.e)
    '''
    # Organise parameters
    try: 
        show_plot = params['show_plot']
    except KeyError:
        show_plot = True

    # Init output list
    method_errors = []
    
    # Loop through the requested methods
    for method in methods:
        # Different h values to evaluate the error with.
        hs = np.logspace(0, -5, 200)
        errors = []

        for h in hs:
            # Solve the ODE from t0 to t1
            t = np.linspace(t0,t1,2)
            X = ode.solve_ode(method, f, t, X0, h_max=h, **params)
            # Calculate error
            if 1:
                error = np.mean(np.abs(X[-1] - X1_true)) # if system of ODEs, take average
            else:
                error = np.abs(X[-1] - X1_true)
            errors.append(error)

        # Plot line for each method
        plt.loglog(hs, errors, label = str(method), linewidth=2)
        method_errors.append(errors)
    
    plt.xlabel('Delta t'), plt.ylabel('Error'), plt.title('Error Plot'), plt.legend()
    if show_plot:
        plt.show()

    return method_errors, hs


