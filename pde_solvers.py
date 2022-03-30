import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

def forward_euler_step(u, t, x, L, BC, BC_type, j, kappa):
    '''
    Function that carries out one step of the forward Euler numerical method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

    t : np.array
        The array containing the times to solve the PDE at.
    
    L : float
        The extent of the space doamin.
    
    BC : function
        The function that calculates the boundary conditions.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann', 'robin' or 'periodic'.
    
    lmbda : float
        The value of lambda computed from mx, mt and the diffusion coefficient.

    j : int
        The index for the time position in the solution.

    Returns
    -------
    u : np.array
        The updated solution matrix.
    '''
    Lmat = construct_L(u, j, BC, L, t, x, kappa, BC_type)

    U_new = (identity(u.shape[0]) + Lmat).dot(u[:,j])

    if BC_type == 'dirichlet':
        u[:, j+1] = boundary_operator(U_new, j+1, BC, L, t, BC_type)
    else:
        u[:, j+1] = boundary_operator(U_new, j, BC, L, t, BC_type)

    return u

def backward_euler_step(u, t, x, L, BC, BC_type, j, kappa):
    '''
    Function that carries out one step of the backward Euler numerical method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

    t : np.array
        The array containing the times to solve the PDE at.
    
    L : float
        The extent of the space doamin.
    
    BC : function
        The function that calculates the boundary conditions.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann', 'robin' or 'periodic'.
    
    lmbda : float
        The value of lambda computed from mx, mt and the diffusion coefficient.

    j : int
        The index for the time position in the solution.

    Returns
    -------
    u : np.array
        The updated solution matrix.
    '''

    if BC_type == 'dirichlet':
        Lmat = construct_L(u, j, BC, L, t, x, kappa, BC_type)
    else:
        Lmat = construct_L(u, j+1, BC, L, t, x, kappa, BC_type)

    U_new = boundary_operator(u[:, j], j+1, BC, L, t, BC_type)
    
    A = (identity(u.shape[0]) - Lmat)

    u[:, j+1] = spsolve(A, U_new)

    return u

def crank_nicholson_step(u, t, x, L, BC, BC_type, j, kappa):
    '''
    Function that carries out one step of the fcrank-nicholson method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, including the boundaries.

    t : np.array
        The array containing the times to solve the PDE at.
    
    L : float
        The extent of the space domain.

    T : float
        The extent of the time domain.
    
    BC : function
        The function that calculates the boundary conditions.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann', 'robin' or 'periodic'.
    
    lmbda : float
        The value of lambda computed from mx, mt and the diffusion coefficient.

    j : int
        The index for the time position in the solution.

    Returns
    -------
    u : np.array
        The updated solution matrix.
    '''

    if BC_type == 'dirichlet':

        Lmat = construct_L(u, j, BC, L, t, x, kappa, BC_type)

        A1 = (identity(u.shape[0]) - 0.5*Lmat)
        A2 = (identity(u.shape[0]) + 0.5*Lmat)

        U_new = A2.dot(u[:,j])

        U_new = boundary_operator(U_new, j+1, BC, L, t, BC_type)

        u[:, j+1] = spsolve(A1, U_new)

    else:
        Lmat1 = construct_L(u, j, BC, L, t, x, kappa, BC_type)
        A1 = (identity(u.shape[0]) + 0.5*Lmat1)

        U_new = A1.dot(u[:,j])

        U_new = boundary_operator(U_new, j, BC, L, t, BC_type, CN=True)

        Lmat2 = construct_L(u, j+1, BC, L, t, x, kappa, BC_type)
        A2 = (identity(u.shape[0]) - 0.5*Lmat2)
        
        u[:,j+1] = spsolve(A2, U_new)

    return u
        

def boundary_operator(u, j, BC, L, t, BC_type, CN = False):
    '''
    Function that applies the effect of the boundary conditions to the solution vector.

    Parameters
    ----------
    u : np.array
        Array containing the solution at timestep j.

    j : int
        The current step in the algorithm.

    BC : function
        The function containing the BC for the PDE..
    
    L : float
        The extent of the space domain.

    t : np.array
        The times at which the PDE is evaluated.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann', 'robin' or 'periodic'.

    CN : bool
        Bool that must be set to true when the method being using is the Crank-Nicholson method.

    Returns
    -------
    Bn : np.array
        The solution vector with the effect of the boundary conditions applied.
    '''
    delta_T = t[1]-t[0] # step size in time
    h = L/(len(u)-1) # step size in space
    Bn = u

    if BC_type == 'dirichlet':
        # Get BC output
        gamma0 = BC(0,t[j])
        gammaL = BC(L,t[j])
        # Apply BC to solution vector
        Bn[0] = gamma0 
        Bn[-1] = gammaL

    if BC_type == 'robin':
        # Get BC output
        gamma0, alpha0 = BC(0,t[j])
        gammaL, alphaL = BC(L,t[j])

        if CN:
            # Get BC for t+1
            gamma02, alpha0 = BC(0,t[j+1])
            gammaL2, alphaL = BC(L,t[j+1])
            # Apply BC to solution
            Bn[0] += (-delta_T/h)*(gamma0 + gamma02)
            Bn[-1] += (-delta_T/h)*(gammaL + gammaL2)

        else:
            # Apply BC to solution
            Bn[0] += (-2*delta_T/h)*gamma0 
            Bn[-1] += (-2*delta_T/h)*gammaL

    if BC_type == 'neumann':
        # Get BC output
        gamma0 = BC(0,t[j])
        gammaL = BC(L,t[j])

        if CN:
            # Get BC for t+1
            gamma02 = BC(0,t[j+1])
            gammaL2 = BC(L,t[j+1])
            # Apply BC to solution
            Bn[0] += (-delta_T/h)*(gamma0 + gamma02)
            Bn[-1] += (-delta_T/h)*(gammaL + gammaL2)

        else:
            # Apply BC to solution
            Bn[0] += (-2*delta_T/h)*gamma0 
            Bn[-1] += (-2*delta_T/h)*gammaL

    return Bn

def construct_L(u, j, robin_BC, L, t, x, kappa, BC_type):
    '''
    Function that constructs the matrix, L, for all of the methods and BC types. L is
    tridiagonal in all cases but the periodic BC case.

    Parameters
    ----------
    u : np.array
        The array containing the solution in both the time and space dimensions.

    j : int
        The current step in the algorithm.

    robin_BC : function
        The function containing the BC for the PDE. Always pass the BC to this function, 
        however, it is only used in the robin BC case.
    
    L : float
        The extent of the space domain.

    t : np.array
        The times at which the PDE is evaluated.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann', 'robin' or 'periodic'.

    Returns
    -------
    L : csr sparse matrix
        The matrix constructed to be used in the step functions.
    '''
    h = x[1] - x[0]
    deltat = t[1] - t[0]

    lmbda1 = deltat/(h**2) * kappa(x + h/2)
    lmbda1 = lmbda1[1:]
    lmbda2 = deltat/(h**2) * (kappa(x+h/2) + kappa(x-h/2))
    lmbda3 = deltat/(h**2) * kappa(x-h/2)
    lmbda3 = lmbda3[:-1]


    a = np.ones(u.shape[0]-1)
    b = np.array([-1]*u.shape[0])
    
    if BC_type == 'dirichlet':
        # Modifies L for dirichlet BCs
        a[-1] = 0
        b[0] = 0
        b[-1] = 0

    elif BC_type == 'neumann':
        # Modifies L for Neumann BCs
        a[-1] = 2

    elif BC_type == 'robin':
        # Modifies L for robin BCs
        a[-1] = 2

        gamma0, alpha0 = robin_BC(0,t[j])
        gammaL, alphaL = robin_BC(L,t[j])

        b[0] = -2*(1+h*alpha0)
        b[-1] = -2*(1+h*alphaL)

    test = lmbda1*a

    L = diags((lmbda1*a, lmbda2*b, lmbda3*np.flip(a)), (-1,0,1), format='csr')

    if BC_type == 'periodic':
        # Modifies L for periodic BCs
        L[-1,0] = lmbda1[0]*1
        L[0,-1] = lmbda3[-1]*1

    return L



def solve_pde(L, T, mx, mt, BC_type, BC, IC, solver, RHS = lambda x,t:0, kappa = lambda x:np.ones(len(x))/10):
    '''
    Function that solves a 1D diffusion equation using the numerical scheme specified.

    Parameters
    ----------
    L : float
        The length of the spatial domain.
    
    T : float
        The maximum time to solve to PDE to.

    mx : int
        The resolution in the spatial dimension to use for the solution.

    mt : int
        The resolution in the time dimension to use for the solution.

    kappa : float
        The diffusion parameter for the PDE.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann', 'robin' or 'periodic'.

    BC : function
        The boundary condition as a callable function. Must take 2 arguments, x and t.
        Can be any function when using BCs of type 'periodic'.

    IC : function
        The initial condition as a callable function. Must take 2 arguments, x and L.

    RHS : function
        The function containing the RHS of the PDE.
    
    solver : string
        The string defining the numerical method to use.
        'feuler' uses the forward Euler scheme.
        'beuler' uses the backwards Euler scheme.
        'cn' uses the Crank-Nicholson scheme.

    Returns
    -------
    u : np.array
        A numpy array containing the solution u for all x and t. The rows hold the spatial
        dimension and the columns hold the time dimension.
    
    t : np.array
        A 1D numpy array containing the values of t that correspond to the columns in u.

    Example
    -------
    u,t = solve_pde(L=1, T=0.5, mx=10, mt=100, kappa=1, 'feuler')
    '''

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    #lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    kappas = kappa(x)
    lmbdas = kappas*deltat/(deltax**2)

    # initialise the solution matrix
    u = np.zeros((x.size, t.size))

    # Check BC_type
    BC_types = ['neumann', 'dirichlet', 'periodic', 'robin']
    if BC_type not in BC_types:
        raise ValueError('Incorrect BC type specified!')

    # Check initial condition
    if not callable(IC):
        raise ValueError('Initial condition specified is not a function!')

    # Check boundary conditions
    if not callable(BC):
        raise ValueError('Boundary condition speicified is not a function!')

    if solver == 'feuler':
        # Checks if solver will be stable with this lambda value
        if (lmbdas > 0.5).any():
            raise ValueError('Lambda greater than 0.5! Consider reducing mx.')
        if (lmbdas < 0).any():
            raise ValueError('Lambda less than 0! Wrong args.')
        solver = forward_euler_step
    elif solver == 'beuler':
        solver = backward_euler_step
    elif solver == 'cn':
        solver = crank_nicholson_step
    else:
        raise ValueError('Solver specified does not exist!')

    # Get initial conditions and apply
    for i in range(len(x)):
        u[i,0] = IC(x[i], L)

    # Remove one boundary for periodic BCs
    if BC_type == 'periodic':
        u = u[0:-1,:]
        x = x[0:-1]
        # BCs have no effect
        BC = lambda x,t:0

    for j in range(0, mt):
        # Carry out solver step, including the boundaries
        u = solver(u, t, x, L, BC, BC_type, j, kappa)
        u[:,j+1] += deltat*RHS(x, t[j])

    return u, t


def robin_BC(x, t):

    # Effect of dirichlet
    alpha = lambda t : 1

    # Effect of neumann
    beta = lambda t : 0

    # The dirichlet and neumann effects on the boundary
    dirichlet_u = lambda t : 1-np.sin(t)
    neumann_u = lambda t : 0

    # Combination of all effects
    gamma = alpha(t)*dirichlet_u(t) + beta(t)*neumann_u(t)

    return gamma, alpha(t)

def RHS1(x, t):
    rhs = np.zeros(len(x))
    rhs += 1

    return rhs


def main():

    # Set problem parameters/functions
    kappa = 0.1   # diffusion constant
    L=1      # length of spatial domain
    T=0.5      # total time to solve for

    # Set numerical parameters
    mx = 100     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    def homo_BC(x, t):
        # Homogeneous BCs, always return 0
        return 0

    # Non-zero Dirichlet BCs
    def non_homo_BC(x,t):
        # Return the sin of the time
        return np.sin(t)

    def u_I(x, L):
        # initial temperature distribution
        y = (np.sin(pi*x/L))
        return y

    def u_I2(x, L):
        if x<0.5:
            return 10
        else:
            return 0

    def u_exact(x,t, kappa, L):
        # the exact solution
        y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
        return y

    def u_exact_nhomo(x,t,kappa,L=1):
        # Exact solution for homogeneous Neumann case (only for L=1)
        y = (2/np.pi) - (4/(3*pi))*np.cos(2*np.pi*x)*(np.e**(-4*(np.pi**2) * kappa * t)) - (4/(15*pi))*np.cos(4*np.pi*x)*(np.e**(-16*(np.pi**2) * kappa * t))
            
        return y

    homo_RHS = lambda x,t : 0

    # Get numerical solution
    u,t = solve_pde(L, T, mx, mt, 'dirichlet', homo_BC, u_I2, solver='feuler', RHS = homo_RHS, kappa = lambda x:np.ones(len(x))*np.sin(np.pi*x)/10)

    # Plot solution in space and time
    from plots import plot_pde_space_time_solution
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map')

    u,t = solve_pde(L, T, mx, mt, 'dirichlet', homo_BC, u_I, solver='beuler', RHS = RHS1, kappa = lambda x:np.ones(len(x))/10)
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map Robin (both homo)')


    # Plot x at time T from the exact solution and numerical method.
    xx = np.linspace(0,L,mx+1)
    from plots import plot_pde_specific_time
    plot_pde_specific_time(u, t, 0.3, L, 'Diffusion Solution', u_exact(xx, 0.3, kappa, L))


    u,t = solve_pde(L, T, mx, mt, 'neumann', non_homo_BC, u_I, solver='feuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map non-homo f')

    u,t = solve_pde(L, T, mx, mt, 'dirichlet', non_homo_BC, u_I, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map diri non-homo cn')


    u,t = solve_pde(L, T, mx, mt, 'neumann', homo_BC, u_I, solver='beuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map homo_neu b')
    plot_pde_specific_time(u, t, 0.5, L, 'Specific Time Feuler Neumann Homo', u_exact_nhomo(xx,0.5,kappa,L=1))

    u,t = solve_pde(L, T, mx, mt, 'periodic', homo_BC, u_I2, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map Periodic cn')

    u,t = solve_pde(L, T, mx, mt, 'neumann', homo_BC, u_I, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map nonhomo_neu cn')


    return 0

if __name__ == '__main__':
    main()