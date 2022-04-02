import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
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
    '''
    Function that carries out natural parameter continuation for a given function and discretisation.

    Parameters
    ----------
    param_list : list
        A list of the parameter values to be used in the continuation.

    sols : list
        A list of solutions. Initially it must only contain the initial solution (initial_u).

    param_to_vary : string
        A string the specifies the parameter that is to be varied in the continuation.

    function : function
        The function containing the equation or system of ODEs to which the continuation method
        will be carried out on.

    discretisation : function
        Function that forms the root finding problem.

    phase_condition : function
        If the system is an ODE system, a phase condition is required to from the root finding
        problem for the numerical shooting process.

    **params:
        Any parameters required to compute the output of the system.

    Returns
    -------
    sols : list
        Array containing the solutions to the system for each parameter value.

    param_list : list
        List containing all the parameter values that have been used to evaluate the system
    '''
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
        return sols, param_list

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
    '''
    Function that carries out natural parameter continuation for a given function and discretisation.

    Parameters
    ----------
    param_list : list
        A list of the parameter values to be used in the continuation.
    
    param_range : list
        A list of length 2 that contains the upper and lower bounds for the parameter
        that is to be varied.

    sols : list
        A list of solutions. Initially it must only contain the initial solution (initial_u).

    param_to_vary : string
        A string the specifies the parameter that is to be varied in the continuation.

    function : function
        The function containing the equation or system of ODEs to which the continuation method
        will be carried out on.

    discretisation : function
        Function that forms the root finding problem.

    Ts : list
        A list containing the computed periods of the orbits. Initially this will have one entry, the
        T_guess supplied to the continuation function.

    phase_condition : function
        If the system is an ODE system, a phase condition is required to from the root finding
        problem for the numerical shooting process.

    **params:
        Any parameters required to compute the output of the system.

    Returns
    -------
    sols : list
        Array containing the solutions to the system for each parameter value.

    param_list : list
        List containing all the parameter values that have been used to evaluate the system
    '''
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



def continuation(initial_u, param_to_vary, param_range, no_param_values, function, method = 'pseudo-arclength', discretisation=None,
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

    discretisation : string
        The chosen discretisation for the numerical continuation. If the continuation is for an equation leave empty,
        if it is for an ODE (system) choose 'numerical-shooting'.

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

    # Select the required discretisation
    if not discretisation:
        discretisation = lambda x:x
    elif discretisation == 'numerical-shooting':
        discretisation = numerical_shooting
    else:
        raise ValueError('Incorrect discretisation entered!')

    # Select and carry out method
    if method == 'pseudo-arclength':
        sols, param_list = pseudo_arclength(param_list, param_range, sols, param_to_vary, function, discretisation, Ts, phase_condition, **params)
    elif method == 'natural-parameter':
        sols, param_list = natural_parameter(param_list, sols, param_to_vary, function, discretisation, phase_condition, **params)
    else:
        raise ValueError('Incorrect method specified!')

    # Remove initial_u from solution vector
    sols = np.array(sols[1:])
    return sols, param_list




def main():

    # ## EXAMPLES
    # 
    # Numerical continuation is the process of finding the location of periodic orbits or equilibria as a parameter in the equation is varied. This Method can be used to find the roots of equations or equilibria and orbits of ODEs and systems of ODEs.
    # 
    # Two methods of numerical continuation are implemented in this package, natural parameter continuationa and pseudo-arclength continuation. Examples of both of these methods are contained within this notebook.


    # Firstly, numerical continuation can be applied to equations, for example, polynomials. The continuation algorithm will 
    # find the location of any roots of the equation as a parameter varies. This can be shown with a cubic equation,
    # 
    # $x^{3} - x + c = 0$,
    # 
    # This equation must first be encoded as a Python function as in the following cell.


    # Function containing cubic equation parameterised by c
    def cubic(x,params):
        c = params['c']
        return x**3 - x + c

    # Numerical continuation can be applied to the function above using the code shown in the cell below. 
    # The parameter, $c$, is to be varied between the values of $-2$ and $2$. The continuation function returns an 
    # array containing the values of $x$ for which roots exist for the corresponding parameter values in the list, returned as p.
    # 
    # Initially, natural parameter continuation will be used as, of the two methods, it is the simpler one.

    # Carry out numerical continuation with natural parameter continuation
    u, p  = continuation([1], 'c', [-2,2], 50, cubic, method='natural-parameter', c=-2)

    plt.plot(p,u, label='Natural')
    plt.xlabel('c'), plt.ylabel('Root Location'), plt.title('Natural Parameter Continuation')
    plt.show()


    # From the graph above, it can be seen that the natural parameter continuation method performs well from values of $c$ between $-2$ and $0.5$. For values greater than this, the algorithm fails to indentify the roots. This is due to the fact that more than one root exists for such values of $c$.
    # 
    # To overcome this, pseudo-arclength continuation can be used, as shown in the following cell.

    # Carry out numerical continuation with pseudo-arclength continuation
    u, p  = continuation([1], 'c', [-2,2], 50, cubic, method='pseudo-arclength', c=-2)

    # Plot the output
    plt.plot(p,u, label='PA')
    plt.xlabel('c'), plt.ylabel('Root Location'), plt.title('Pseudo-Arclength Continuation')
    plt.show()

    # The plot above shows that the pseudo-arclength method performs much more gracefully when the curve of roots turns a `corner'.
    # 
    # The more common use of numerical continuation is to plot out the positions of periodic orbits for certain parameter 
    # values for systems of ODEs. This is equivalent to plotting parts of the bifurcation diagram resulting from the system of ODEs.
    # 
    # An example of this will be demonstrated below, using a modified version of the Hopf Bifurcation normal form, where the parameter to be varied is $\beta$,
    # 
    # $\frac{du_{1}}{dt} = \beta u_{1} - u_{2} + u_{1}(u_{1}^{2} + u_{2}^{2}) - u_{1}(u_{1}^{2} + u_{2}^{2})^{2}$,
    # 
    # $\frac{du_{2}}{dt} =  u_{1} + \beta u_{2} + u_{1}(u_{1}^{2} + u_{2}^{2}) - u_{2}(u_{1}^{2} + u_{2}^{2})^{2}$,
    # 
    # which can be encoded as a Python function as shown in the following cell.

    def modified_hopf(X, t, params):
        '''
        Function containing the modified Hopf normal form system of ODEs.

        Parameters
        ----------
        X : np.array
            Array containing the system state variables, u1 and u2.
        t : float
            The time to evaluate the system at.
        params : dict
            Dictionary containing the parameter requires for this system (beta).

        Returns
        -------
        X : list
            List containing the gradients with respect to time for u1 and u2.
        '''

        beta = params['beta']

        u1 = X[0]
        u2 = X[1]

        du1dt = beta*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
        du2dt = u1 + beta*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2

        X = [du1dt, du2dt]

        return X

    # When applying numerical continuation to ODEs or systems of ODEs, the periodic orbits are found using numerical shooting, 
    # therefore, it is necessay to define a phase condition to fix the position of the orbits found. The phase condition, 
    # 
    # $\frac{du_{1}(t=0)}{dt}=0$,
    # 
    # can be encoded as a Python function as in the following cell.

    def pc_modified_hopf(X0, **params):
        '''
        Function containing the phase condition described above for the modified Hopf ODE system

        Parameters
        ----------
        X0 : np.array
            The proposed intial conditions of a periodic orbit.
        **params:
            Any parameters required to evaluate the system.

        Returns
        -------
        pc : float
            The gradient of u1 with respect to t at t=0.
        '''
        # returns du1dt at t=0 (to be set =0)
        pc = modified_hopf(X0, 0, params)[0]
        return pc

    # Initially using natural parameter continuation, the orbits can be found for various values of $\beta$ as follows,

    # Carry out the numerical continuation using natural parameter continuation
    u, p =  continuation([1,1], 'beta', [2,-1], 50, modified_hopf, 'natural-parameter', 'numerical-shooting', phase_condition=pc_modified_hopf, T_guess=6, beta=-1)

    # Plot the output
    plt.plot(p,u[:,0], label='PA')
    plt.xlabel('beta'), plt.ylabel('u1'), plt.title('Natural Parameter Continuation')
    plt.show()

    # Similarly to the cubic equation, natural parameter continuation struggles when there is a `corner'. 
    # Therefore, the more robust pseudo-arclength method can be used as shown in the following cell.

    # Carry out the numerical continuation using pseudo-arclength continuation
    u, p =  continuation([1,1], 'beta', [2,-1], 50, modified_hopf, 'pseudo-arclength', 'numerical-shooting', phase_condition=pc_modified_hopf, T_guess=6, beta=-1)

    # Plot the output
    plt.plot(p,u[:,0], label='PA')
    plt.xlabel('beta'), plt.ylabel('u1'), plt.title('Pseudo-Arclength Continuation')
    plt.show()

    # The methods described here can be used for equations and systems of ODEs of arbitrary dimensions.




if __name__ == '__main__':
    main()