import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ode import hopf, modified_hopf
from numerical_shooting import numerical_shooting, root_finding_problem


def get_arc(u3, u2, u1, param3, param2, param1):
    '''
    Function that returns the dot product between the linear prediction of the next point in the
    numerical continuation and the vector between the prediction and the proposed next point provided
    by the root finding algorithm. This is the pseudo-arclength condition.

    Parameters
    ----------
    u3 : np.array
        Current state vector that is being solved for by the root finding method.
    
    u2 : np.array
        State vector computed for iteration t-1 (where t is the current iteration).

    u1 : np.array
        State vector computed for iteration t-2 (where t is the current iteration).
    
    param3 : float
        Current parameter value that is being solved for by the root finding method.

    param2 : float
        Parameter value for iteration t-1 (where t is the current iteration).

    param1 : float
        Parameter value for iteration t-2 (where t is the current iteration).
    
    Returns
    -------
    arc : float
        The dot product to be set to 0 by the root finding method. It is the dot product between
        the secant and the vector joining [u3,p3] and [u2,p2].
    '''
    # Calculate the (linear prediction of next (u,p))
    pred = np.append(u2 + (u2 - u1), param2 + (param2 - param1))
    # Create vector of the proposed correction
    target = np.append(u3, param3)
    # Calculate secant
    secant = np.append((u2-u1), (param2-param1))

    # Take the dot product to test for orthogonality
    arc = np.dot(target - pred, secant)

    return arc
        


def root_finding(x, discretisation, function, u1, u2, p1, p2, param_to_vary, phase_condition, params):
    '''
    Function that forms the root finding problem to be solved by the pseudo-arclength numerical
    continuation method.
    
    Parameters
    ----------

    x : np.array
        Array containing the current guess at the solution to the root finding problem. This contains
        the system state variables augmented with the current parameter value.
    
    discretisation : function
        For an ODE this is the numerical shooting function (numerical_shooting).
        For some function (not ODE) this is simply lambda x:x which is the default.

    function : function
        The function containing the system the numerical continuation is being applied to.

    u1 : np.array
        State vector computed for iteration t-2 (where t is the current iteration).

    u2 : np.array
        State vector computed for iteration t-1 (where t is the current iteration).

    param1 : float
        Parameter value for iteration t-2 (where t is the current iteration).

    param2 : float
        Parameter value for iteration t-1 (where t is the current iteration).

    param_to_vary : string
        The name of the parameter that is being varied in the numerical continuation.

    phase_condition : function
        If the system is an ODE sysetm, a phase condition is required to create the root finding
        problem. Pass None value is system is not an ODE system.

    params : dict
        The parameters required to solve the system, including the parameter that is
        being varied.

    Returns
    -------
    root : np.array
        The combination of the discretisation output and the pseudo-arclength condition.
        This quantity is being solved for by the root finder.
    '''
    # Unpack the variables from the function input
    u0 = x[:-1]
    p0 = x[-1]
    # Set the parameter value to be passed to the function
    params[param_to_vary] = p0

    # If the problem involves an ODE
    if discretisation == numerical_shooting:
        d = root_finding_problem(list(u0), function, phase_condition, params)
    # If the problem involves a non-ODE equation
    else:
        d = discretisation(function(u0, params))
    # Add the pseudo-arclength condition    
    arc = get_arc(u0, u2, u1, p0, p2, p1)
    root = np.append(d, arc)
    
    return root




def natural_parameter(param_list, sols, param_to_vary, function, discretisation, phase_condition, **params):

    T_guess = 5
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
    
    return sols, param_list


def pseudo_arclength(param_list, param_range, sols, param_to_vary, function, discretisation, Ts, phase_condition, **params):

    param_list = list(param_list[:2])
    initial_u = sols[-1]
     # While the parameter lies within the specified range, continue
    while np.min(param_range) <= param_list[-1] and param_list[-1] <= np.max(param_range):
        
        # Generate the first 2 system states and parameter values
        if len(sols)==1:
            for i in range(2):
                # Set parameter
                param1 = param_list[-2+i]
                params[param_to_vary] = param1
                # If ODE system
                if discretisation == numerical_shooting:
                    X0, T = numerical_shooting(sols[-1].copy(), Ts[-1], function, phase_condition, **params)
                    u1 = list(X0)
                    Ts.append(T)
                # If non-ODE system
                else:
                    u1 = fsolve(discretisation(function), initial_u, args=params)
                sols.append(list(u1))

        # Get system states and param values for next continuation step      
        u1 = np.array(sols[-2])
        u2 = np.array(sols[-1])
        param1 = param_list[-2]
        param2 = param_list[-1]
    
        # Add period to state vector is searching for orbits
        if discretisation == numerical_shooting:
            u1 = np.append(u1, Ts[-2])
            u2 = np.append(u2, Ts[-1])
            pred = np.append(u2 + (u2 - u1), param2+(param2-param1)) 
        else:
            pred = np.append(u2 + (u2 - u1),param2+(param2-param1))

        # Solve root finding problem and store the results
        x = fsolve(root_finding, np.array(pred),
                    args = (discretisation, function, u1, u2, param1, param2, param_to_vary, phase_condition, params))

        sols.append(list(x[:len(initial_u)]))
        Ts.append(x[-2])
        param_list.append(x[-1])
    
    return sols, param_list



def continuation(initial_u, param_to_vary, param_range, no_param_values, function, method = 'pseudo-arclength', discretisation=lambda x:x,
                                                phase_condition = None, T_guess = 5, **params):
    '''
    Function that carries out the numerical continuation either using natural parameter continuation
    or pseudo-arclength continuation.

    Parameters
    ----------
    initial_u : list
        List containing the initial conditions for the system.

    param_to_vary : string
        The name of the parameter to be varied in the numerical continuation.

    param_range : list
        List of length 2 that contains the bounds within which the parameter is to be varied.

    no_param_values : int
        The number of different parameter values to solve the system for. This controls the
        resolution of the output continuation.

    function : function
        Function containing the system the continuation is to be applied to.

    method : string
        The method to use for numerical continuation. Either 'natural-parameter' or 'pseudo-arclength'.

    discretisation : function
        Function that forms the root finding problem along with the supplied system and
        the pseudo-arclength condition.

    phase_condition : function
        If the system is an ODE system, a phase condition is required to from the root finding
        problem for the numerical shooting process.

    T_guess : float
        Initial guess for the period of the orbits being found.

    **params:
        Any parameters required to compute the output of the system.

    Returns
    -------
    sols : np.array
        Array containing the solutions to the system for each parameter value.

    param_list : list
        List containing all the parameter values that have been used to evaluate the system.

    Example
    -------
        u, p = continuation([-1,-1], 'beta', [2,0], 50, hopf, 'pseudo-arclength', numerical_shooting,
                                                phase_condition=pc_hopf, T_guess=6, beta=-1, sigma=-1)
    '''
    # Form param list and retrieve first 2 entries to begin continuation
    param_list = np.linspace(param_range[0], param_range[1], no_param_values)

    # Save the initial system state and initial guess at orbit period.
    sols = [initial_u]
    Ts = [T_guess]

    # Select and carry out method
    if method == 'pseudo-arclength':
        pseudo_arclength(param_list, param_range, sols, param_to_vary, function, discretisation, Ts, phase_condition, **params)
    elif method == 'natural-parameter':
        natural_parameter(param_list, sols, param_to_vary, function, discretisation, phase_condition, **params)
    else:
        raise ValueError('Incorrect method specified!')

    # Remove initial_u from solution vector
    sols = np.array(sols[1:])
    return sols, param_list




def main():

    # Function containing cubic equation parameterised by c
    def cubic(x,params):
        c = params['c']
        return x**3 - x + c

    # Phase condition for hopf normal form
    def pc_hopf(X0, **params):
        # returns du1dt at t=0 (to be set =0)
        return hopf(X0, 0, params)[0]

    # Phase condition for modified hopf
    def pc_modified_hopf(X0, **params):
        # returns du1dt at t=0 (to be set =0)
        return modified_hopf(X0, 0, params)[0]


    # Testing continuation with the cubic equation
    u, p  = continuation(1, 'c', [-2,2], 20, cubic, method='natural-parameter', c=-2)
    plt.plot(p,u, label='Natural')
    u, p = continuation([1], 'c', [-2,2], 50, cubic, method='pseudo-arclength', c=-2)
    plt.plot(p,u,label='Pseudo-Arclength')
    plt.xlabel('C'), plt.ylabel('Root Location'), plt.title('Pseudo-Arclength Continuation')
    plt.show()

    
    # Testing continuation with the hopf normal form
    u, p =continuation([-1,-1], 'beta', [2,0], 50, hopf, 'natural-parameter', numerical_shooting, phase_condition=pc_hopf, beta=-1, sigma=-1)
    plt.plot(p,u[:,0], label='Natural')
    u, p = continuation([-1,-1], 'beta', [2,0], 50, hopf, 'pseudo-arclength', numerical_shooting, phase_condition=pc_hopf, T_guess=6, beta=-1, sigma=-1)
    plt.plot(p,u[:,0], label='Pseudo-Arclength')
    plt.xlabel('Beta'), plt.ylabel('u'), plt.title('Continuation with Hopf'), plt.legend()
    plt.show()

    # Testing continuation with modified hopf normal form
    u, p =  continuation([1,1], 'beta', [2,-1], 20, modified_hopf, 'natural-parameter', numerical_shooting, phase_condition=pc_modified_hopf, T_guess=6, beta=-1)
    plt.plot(p,u[:,0], label='Natural')
    u, p = continuation([1,1], 'beta', [2,-1], 100, modified_hopf, 'pseudo-arclength', numerical_shooting, phase_condition=pc_modified_hopf, T_guess=6, beta=2)
    plt.plot(p,u[:,0], label='Pseudo-Arclength')
    plt.xlabel('Beta'), plt.ylabel('u'), plt.title('Continuation with Modified Hopf'), plt.legend()
    plt.show()
    

    return 0


if __name__ == '__main__':
    main()