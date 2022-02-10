import matplotlib.pyplot as plt


def plot_solution(t, X, X_true=None):
    '''
    Function that plots the numerical solution to an ODE as well as the true solution
    if required.

    ARGS:   t = the time over the ODE has been solved.
            X = the numerical solution at each t_i in t.
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
            plt.title('Approx and True Solution to ODE'), plt.xlabel('t'), plt.legend()

        # No X_true provided, just plot X
        else:
            for i in range(number_of_vars):
                plt.plot(t, X[:,i], label='Approx Solution, x'+str(i))
            plt.title('Approx Solution to ODE'), plt.xlabel('t'), plt.legend()

    else:
        number_of_vars = 1
        # If true solution is provided plot both X and X_true
        if X_true is not None:
            plt.plot(t, X_true[:], label='True Solution')
            plt.plot(t, X[:], label='Approx Solution')
            plt.title('Approx and True Solution to ODE'), plt.xlabel('t'), plt.legend()

        # No X_true provided, just plot X
        else:
            plt.plot(t, X[:], label='Approx Solution')
            plt.title('Approx Solution to ODE'), plt.xlabel('t'), plt.legend()

    
    
    plt.show()