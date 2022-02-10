import matplotlib.pyplot as plt
import numpy as np
from ode_solvers import *


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


def plot_error(methods, f, t0, t1, X0, X1_true):

    method_errors = []

    for method in methods:
        hs = np.logspace(-1, -5, 5)
        errors = []

        for h in hs:
            print(h)
            t = np.arange(t0,t1,h)
            X = solve_ode(method, f, t, X0, h_max=1)

            error = np.abs(X[-1] - X1_true)
            errors.append(error)
            
        method_errors.append(errors)

    i=0
    for errs in method_errors:
        plt.loglog(hs, errs, label = str(methods[i]))
        i+=1
    
    plt.xlabel('Delta t'), plt.ylabel('Error'), plt.title('Error Plot'), plt.legend()
    plt.show()

    return 0