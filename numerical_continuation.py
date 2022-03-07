import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ode import hopf, modified_hopf
from numerical_shooting import numerical_shooting


def cubic(x,params):
    c = params['c']
    return x**3 - x + c


def natural_param_continuation(initial_u, param_to_vary, param_range, no_param_values, function, discretisation=lambda x:x,
                                    solver='fsolve', phase_condition = None, T_guess = 5, **params):
    '''
    Function that uses the natural parameter continuation method.

    Parameters
    ----------
    initial_u : list
        Initial guess of the function output (equilibrium or limit cycle).

    param_to_vary : string
        The name of the parameter that is going to be varied.

    param_range : list
        List containing the start and end points of the varying parameter in the form [a,b]
        where a is the initial parameter value and b is the final value.

    function : function
        Either the function to be continued or the ODE.

    discretisation : function
        For an ODE this is the numerical shooting function.
        For some function (not ODE) this is simply lambda x:x which is the default.

    solve : function
        The root finder to use, default is SciPy's fsolve.
    '''

    # Set up range of parameters for continuation
    param_list = np.linspace(param_range[0], param_range[1], no_param_values)
    sols = [initial_u]

    if discretisation == numerical_shooting:

        for i in range(len(param_list)):
            # Set the value of the varying parameter
            params[param_to_vary] = param_list[i]
            prev_sol = sols[i]
            try:
                X0, T = numerical_shooting(prev_sol.copy(), T_guess, function, phase_condition, **params)
                sols.append(list(X0))
            except ValueError:
                print('Root finder did not converge, try different T_guess')
                break

        # Remove initial u
        sols = sols[1:]
        return np.array(sols), param_list

    # If not an ODE and shooting is not required
    else:
        # Loop through param values
        for i in range(len(param_list)):
            # Set value of varying parameter
            params[param_to_vary] = param_list[i]
            # Solve root finding problem
            root = fsolve(discretisation(function), sols[i], args=params)
            sols.append(root)

        
        return sols[1:], param_list



def main():
    # Testing continuation with the cubic equation
    u, p  = natural_param_continuation(1, 'c', [-2,2], 20, cubic, c=-2)
    plt.plot(p,u)
    plt.show()



    def pc_hopf(X0, **params):
        # returns du1dt at t=0 (to be set =0)
        return hopf(X0, 0, params)[0]

    def pc_modified_hopf(X0, **params):
        # returns du1dt at t=0 (to be set =0)
        return modified_hopf(X0, 0, params)[0]

    # Testing continuation with the hopf normal form
    u, p = natural_param_continuation([1,1], 'beta', [2,0], 50, hopf, numerical_shooting, phase_condition=pc_hopf, beta=-1, sigma=-1)
    plt.plot(p,u[:,0])
    plt.show()

    u, p = natural_param_continuation([1,1], 'beta', [2,-1], 20, modified_hopf, numerical_shooting, phase_condition=pc_modified_hopf, T_guess=6, beta=-1)
    plt.plot(p,u[:,0])
    plt.show()

    return 0




if __name__ == '__main__':
    main()