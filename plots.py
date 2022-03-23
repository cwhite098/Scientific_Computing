import matplotlib.pyplot as plt
import numpy as np
import ode_solvers as ode
from ode import *


def plot_solution(t, X, xlabel='t', ylabel='x', title='Solution', X_true=None):
    '''
    Function that plots the numerical solution to an ODE as well as the true solution
    if required.

    Parameters
    ----------
    t : np.array
        The time over the ODE has been solved.
    X : np.array 
        The numerical solution at each t_i in t.
    xlabel : string 
        The label for the x-axis.
    ylabel : string
        The label for the y-axis.
    title : string 
        The title for the plot.
    X_true : List 
        The true solution at each t_i in t.
    
    Example
    -------    
    plot_solution(t=linspace(0,1,11), X, X_true)
    '''

    # Check to see how many variables are to be plotted
    if len(X.shape) > 1:
        number_of_vars = X.shape[1]
        # If true solution is provided plot both X and X_true
        if X_true is not None:
            X_true = np.array(X_true).transpose()
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
            X_true = np.array(X_true).transpose()
            plt.plot(t, X_true[:], label='True Solution')
            plt.plot(t, X[:], label='Approx Solution')

        # No X_true provided, just plot X
        else:
            plt.plot(t, X[:], label='Approx Solution')
    
    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel), plt.legend()
    plt.show()



def plot_error(methods, f, t0, t1, X0, X1_true,  **params):
    '''
    Function that works out the global error for multiple numerical methods used to solve ODEs
    and plots the error against the step size on a loglog graph.

    Parameters
    ----------
    method : list
        List of methods to plot on the loglog plot.
    f : function
        The function containg the ODE that is to be evaluated.
    t0 : float
        The initial time for the solution.
    t1 : float
        The final time of the solution, this is where the error is calculated.
    X0 : list 
        The initial conditions of the ODE.
    X1_true : list
        The true value of the ODE's solution at t=t1.
    **params:   
        show_plot : bool 
            Boolean that controls whether the plot is shown or not.
        Any parameters necessary for the ODE being solved.

    Returns
    -------
    method_errors : nested list
        The errors at t1 of all the methods for differnt h values.

    Example
    -------
    plot_error(['rk4', 'euler'], f, t0=0, t1=1, X0=1, X1_true=np.e)
    '''
    # Organise parameters
    try: 
        show_plot = params['show_plot']
    except KeyError:
        show_plot = True

    # Init output list
    method_errors = []

    X1_true = np.array(X1_true).transpose()
    
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


def plot_pde_space_time_solution(u, L, T, title):
    '''
    Function that plots a 2 dimensional solution to a pde (space and time) as a heatmap.

    Parameters
    ----------
    u : np.array
        A 2D array containing the solution to the PDE. The rows are the spatial dimension and the
        columns are the time dimension.

    L : float
        The upper limit of the space domain.

    T : float
        The time the solution is calculated until.

    title : string
        The title for the plot.

    Example
    -------
    plot_pde_space_time_solution(u, 1, 0.5, 'Diffusion Equation Solution')
    '''
    # Get the extent of the space and time domains so the plots reflect
    # the values provided to the pde solver.
    extent = [0 , L, 0 , T]
    plt.imshow(u, aspect='auto', extent=extent)
    
    # Set axis labels and title
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.title(title)
    # Show the plot
    plt.show()


def plot_pde_specific_time(u, t, t_plot, L, title, exact = None):
    '''
    Function that plots a slice of the solution to a PDE at a specific time. Works for PDEs with one spatial and 
    one temporal dimensions.

    Parameters
    ----------
    u : np.array
        A 2D array containing the solution to the PDE. The rows are the spatial dimension and the
        columns are the time dimension.

    t : np.array
        A 1D array containing the times that the solution is evaluated at. This is returned by the
        solve_pde function.

    t_plot : float
        The time at which to plot the solution.

    L : float
        The upper limit of the space domain.

    title : string
        The title for the plot.

    exact : np.array
        An array containing the exact solution to the pde at time t=t_plot.
        It must be discretised in the spatial dimension in the same way as the numerical
        solution.

    Example
    -------
    plot_pde_specific_time(u, t, 0.3, L, 'Diffusion Solution', u_exact)
    '''
    # Get the index of the time we want to plot
    index = [t==t_plot]
    if not index:
        raise ValueError('Non-existant time provided!')

    # Select the slice of the solution at t=t_plot
    solution = u[:,index[0]]
    xx = np.linspace(0,L,u.shape[0])
    # Plot u at t=t_plot
    plt.plot(xx, solution, 'o', c='r', label='Numerical Solution')

    # If exact solution provided, plot it as well.
    if not exact.any() == None:
        plt.plot(xx, exact, c='b', label='Exact Solution')

    # Configure the plot and show it
    plt.title(title+' at time t='+str(t_plot))
    plt.xlabel('x'), plt.ylabel('u')
    plt.legend()
    plt.show()


