import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ode import hopf, modified_hopf
from numerical_shooting import numerical_shooting, root_finding_problem


def cubic(x,params):
    c = params['c']
    return x**3 - x + c


def natural_param_continuation(initial_u, param_to_vary, param_range, no_param_values, function, discretisation=lambda x:x,
                                    solver='fsolve', phase_condition = None, T_guess = 5, **params):
    '''
    Function that uses the natural parameter continuation method between two provided
    parameter values. This can either be applied to any function or an ODE with a phase
    condition required to formulate the shooting problem.

    Parameters
    ----------
    initial_u : list
        Initial guess of the function output (equilibrium or limit cycle).

    param_to_vary : string
        The name of the parameter that is going to be varied.

    param_range : list
        List containing the start and end points of the varying parameter in the form [a,b]
        where a is the initial parameter value and b is the final value.

    no_param_values : int
        Integar that defined the number of steps between the upper and lower bound of
        the varying parameter.

    function : function
        Either the function to be continued or the ODE.

    discretisation : function
        For an ODE this is the numerical shooting function (numerical_shooting).
        For some function (not ODE) this is simply lambda x:x which is the default.
    
    solver : function
        The root finder to be used for the continuation. The default is SciPy's
        fsolve.

    phase_condition : function
        When numerically continuing an ODE, a phase condition is needed for the
        shooting problem.

    T_guess : float
        An initial guess of the period of any limit cycles found by the shooting
        operation. The default is 5.

    **params:
        Any parameters needed for the equaiton/ODE, including the parameter to be
        varied.

    Returns
    -------
    sols : list
        A list containing the system state (equilibrium/limit cycle) for each parameter
        value.

    param_list : list
        A list containing the parameter values used for the numerical continuation.

    Example
    -------
    u, p = natural_param_continuation([-1,-1], 'beta', [2,0], 50, hopf, numerical_shooting,
                                        phase_condition=pc_hopf, beta=-1, sigma=-1)
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



def get_arc(u3, u2, u1, param3, param2, param1):

    pred = [u2 + (u2 - u1), param2 + (param2 - param1)]
    secant = [(u2-u1), (param2-param1)]

    arc = np.dot(u3 - pred[0], secant[0]) + np.dot(param3 - pred[1], secant[1])

    return arc
        
def root_finding(x, discretisation, function, u1, u2, p1, p2, param_to_vary, phase_condition, T_guess,  params):
    
    u0 = x[:-1]
    p0 = x[-1]
    params[param_to_vary] = p0
    if discretisation == numerical_shooting:
        U0 = np.append(u0, T_guess)
        d = root_finding_problem(list(U0), function, phase_condition, params)
        d = d[:-1]
        T_guess = d[-1]
    else:
        d = discretisation(function(u0, params))

    arc = get_arc(u0, u2, u1, p0, p2, p1)

    root = np.append(d, arc)
    
    return root




def pseudo_arclength_continuation(initial_u, param_to_vary, param_range, no_param_values, function, discretisation=lambda x:x,
                                    solver='fsolve', phase_condition = None, T_guess = 5, **params):

    # Form param list and retrieve first 2 entries to begin continuation
    param_list = np.linspace(param_range[0], param_range[1], no_param_values)
    param_list = list(param_list[:2])
    sols = [initial_u]
    counter=0

    # Figure out this bool thing
    while counter <= no_param_values:
        counter+=1
        if len(sols)==1:
            for i in range(2):
                param1 = param_list[-2+i]
                params[param_to_vary] = param1
                if discretisation == numerical_shooting:
                    X0, T = numerical_shooting(sols[-1].copy(), T_guess, function, phase_condition, **params)
                    u1 = list(X0)
                else:
                    u1 = fsolve(discretisation(function), initial_u, args=params)
                sols.append(list(u1))
            u1 = np.array(sols[-2])
            u2 = np.array(sols[-1])
            param1 = param_list[-2]
            param2 = param_list[-1]
        else:
            u1 = np.array(sols[-2])
            u2 = np.array(sols[-1])
            param1 = param_list[-2]
            param2 = param_list[-1]

        

        pred = np.append(u2 + (u2 - u1), param2 + (param2 - param1))

        x = fsolve(root_finding, np.array(pred),
                    args = (discretisation, function, u1, u2, param1, param2, param_to_vary, phase_condition, T_guess, params))

        
        sols.append(list(x[:-1]))
        param_list.append(x[-1])

        
    return np.array(sols[1:]), param_list


def main():
    # Testing continuation with the cubic equation
    u, p  = natural_param_continuation(1, 'c', [-2,2], 20, cubic, c=-2)
    plt.plot(p,u)
    plt.xlabel('C'), plt.ylabel('Root Location'), plt.title('Natural Param Continuation')
    plt.draw()

    u, p = pseudo_arclength_continuation(1, 'c', [-2,2], 50, cubic, c=-2)
    plt.plot(p,u)
    plt.xlabel('C'), plt.ylabel('Root Location'), plt.title('Pseudo-Arclength Continuation')
    plt.show()

    def pc_hopf(X0, **params):
        # returns du1dt at t=0 (to be set =0)
        return hopf(X0, 0, params)[0]

    def pc_modified_hopf(X0, **params):
        # returns du1dt at t=0 (to be set =0)
        return modified_hopf(X0, 0, params)[0]

    # Testing continuation with the hopf normal form
    u, p = natural_param_continuation([-1,-1], 'beta', [2,0], 50, hopf, numerical_shooting, phase_condition=pc_hopf, beta=-1, sigma=-1)
    plt.plot(p,u[:,0])
    plt.show()

    u, p = pseudo_arclength_continuation([-1,-1], 'beta', [2,0], 50, hopf, numerical_shooting, phase_condition=pc_hopf, T_guess=6, beta=-1, sigma=-1)
    plt.plot(p,u[:,0])
    plt.show()



    u, p = natural_param_continuation([1,1], 'beta', [2,-1], 20, modified_hopf, numerical_shooting, phase_condition=pc_modified_hopf, T_guess=6, beta=-1)
    plt.plot(p,u[:,0])
    plt.show()

    return 0




if __name__ == '__main__':
    main()