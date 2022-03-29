from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags, vstack, hstack, csr_matrix
from scipy.sparse.linalg import spsolve



def forward_euler_step(u, A, t, L, BC, BC_type, lmbda, j):
    '''
    Function that carries out one step of the forward Euler numerical method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

    A : scipy sparse matrix
        A scipy sparse matrix that is used to calculate the next time step of the solution.

    t : np.array
        The array containing the times to solve the PDE at.
    
    L : float
        The extent of the space doamin.
    
    BC : function
        The function that calculates the boundary conditions.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann' or 'periodic'.
    
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
        # Carry out matrix multiplication
        u[1:-1,j+1] = A.dot(u[1:-1,j])

        # Get BCs @ t=j
        pj = BC(0, t[j])
        qj = BC(L, t[j])
        # Add the boundary itself
        u[0,j] = pj
        u[-1,j] = qj
        # Add effect of boundary to inner rows
        u[1,j+1] += lmbda*pj
        u[-2,j+1] += lmbda*qj

    if BC_type == 'neumann':

        u[:,j+1] = A.dot(u[:,j])
        u[0,j+1] += 2*lmbda*(L/u.shape[0])*(-BC(0,t[j]))
        u[-1,j+1] += 2*lmbda*(L/u.shape[0])*(BC(L,t[j]))

    if BC_type == 'periodic':
        u[:, j+1] = A.dot(u[:, j])
        # Wrap around boundary
        #u[-1,j+1] = u[0,j+1]

    return u

def backward_euler_step(u, A, t, L, BC, BC_type, lmbda, j):
    '''
    Function that carries out one step of the backward Euler numerical method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

    A : scipy sparse matrix
        A scipy sparse matrix that is used to calculate the next time step of the solution.

    t : np.array
        The array containing the times to solve the PDE at.
    
    L : float
        The extent of the space doamin.
    
    BC : function
        The function that calculates the boundary conditions.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann' or 'periodic'.
    
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
 
        # Get modifier for u
        zeros1 = np.zeros(u.shape[0]-2)
        zeros1[0] += lmbda * BC(0, t[j+1])
        zeros1[-1] += lmbda * BC(L, t[j+1])
        # Solve the matrix equation
        u[1:-1,j+1] = spsolve(A, u[1:-1,j] + zeros1)
        # Add the boundary conditions
        u[0,j+1] = BC(0, t[j+1])
        u[-1,j+1] = BC(L, t[j+1])

    if BC_type == 'neumann':
        
        # Add effect over the boundary
        zeros1 = np.zeros(u.shape[0])
        zeros1[0] += 2 * lmbda * L/u.shape[0] * BC(0,t[j])
        zeros1[-1] += 2 * lmbda * L/u.shape[0]* BC(L,t[j])
        # Solve
        u[:,j+1] = spsolve(A, u[:,j] + zeros1)

    if BC_type == 'periodic':
        u[:, j+1] = spsolve(A, u[:, j])

    return u

def crank_nicholson_step(u, A, B, t, L, T, BC, BC_type, lmbda, j):
    '''
    Function that carries out one step of the fcrank-nicholson method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, including the boundaries.

    A : scipy sparse matrix
        A scipy sparse matrix (tri-diagonal) that is used to calculate the next time step of the solution.

    B : scipy sparse matrix
        A scipy sparse matrix (tri-diagonal) that is used to multiply the previous step of the solution.

    t : np.array
        The array containing the times to solve the PDE at.
    
    L : float
        The extent of the space domain.

    T : float
        The extent of the time domain.
    
    BC : function
        The function that calculates the boundary conditions.

    BC_type : string
        The type of boundary conditions, either 'dirichlet', 'neumann' or 'periodic'.
    
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
        # https://mathonweb.com/resources/book4/Heat-Equation.pdf
        R = B.dot(u[:,j])
        # Add BC at x=0 and x=L
        R[0] = BC(0,t[j])
        R[-1] = BC(L,t[j])
        R = csr_matrix(R)

        # Solve for next time step
        u[:,j+1] = spsolve(A, R.transpose())
        
    if BC_type == 'neumann':

        R = B.dot(u[:,j])
        R[0] += T/u.shape[1]*(BC(0,t[j+1]) + BC(0,t[j]))
        R[-1] += T/u.shape[1]*(BC(L,t[j+1]) + BC(L,t[j]))
        R = csr_matrix(R)

        # Solve for next time step
        u[:,j+1] = spsolve(A, R.transpose())

    if BC_type == 'periodic':
        R = B.dot(u[:,j])
        R = csr_matrix(R)

        # Solve for next time step
        u[:,j+1] = spsolve(A, R.transpose())

    return u


def get_matrix(lmbda, BC_type, u, solver):
    '''
    Function that takes the numerical solver being used and the type of boundary conditions
    and returns the matrix or matrices required to carry out the algorithm.
    
    Parameters
    ----------
    lmbda : float
        The value of lambda computed from mx, mt and the diffusion coefficient.

    BC_type : string
        The type of boundary conditions, either 'dirichlet' or 'neumann'.

    u : np.array
        A numpy array containing the solution of the PDE, including the boundaries.

    solver : function
        The step solver to be used. Forwards/Backwards Euler or Crank-Nicholson.

    Returns
    -------
    A, B : np.array
        Matrices required to carry out the numerical PDE approximations.
    '''
    # Matrices for forward Euler
    if solver == forward_euler_step:

        if BC_type == 'dirichlet':
            a = np.array([lmbda]*(u.shape[0]-3))
            b = np.array([1-2*lmbda]*(u.shape[0]-2))
            # Get diagonal matrix
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

        if BC_type == 'neumann':
            a = np.array([lmbda]*(u.shape[0]-1))
            b = np.array([1-2*lmbda]*(u.shape[0]))
            a[-1] = a[-1]*2
            # Get diagonal matrix
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

        if BC_type == 'periodic':
            a = np.array([lmbda]*(u.shape[0]-2))
            b = np.array([1-2*lmbda]*(u.shape[0]-1))
            # Get diagonal matrix
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')
            A[-1,0] = lmbda
            A[0,-1] = lmbda
        return A


    if solver == backward_euler_step:

        if BC_type == 'dirichlet':
            a = np.array([-lmbda]*(u.shape[0]-3))
            b = np.array([1+2*lmbda]*(u.shape[0]-2))
            # Get diagonal matrix
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

        if BC_type == 'neumann':
            a = np.array([-lmbda]*(u.shape[0]-1))
            b = np.array([1+2*lmbda]*(u.shape[0]))
            a[-1] = a[-1]*2
            # Get diagonal matrix
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

        if BC_type == 'periodic':
            a = np.array([-lmbda]*(u.shape[0]-2))
            b = np.array([1+2*lmbda]*(u.shape[0]-1))
            # Get diagonal matrix
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')
            A[-1,0] = -lmbda
            A[0,-1] = -lmbda
        return A
        

    
    if solver == crank_nicholson_step:

        if BC_type == 'dirichlet':
            # Construct the 2 tri-diags needed for the CN scheme
            a = np.array([-(lmbda/2)]*(u.shape[0]-1))
            b = np.array([1+lmbda]*(u.shape[0]))
            b[0] = 1
            b[-1] = 1
            a[-1] = 0
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

            a = np.array([lmbda/2]*(u.shape[0]-1))
            b = np.array([1-lmbda]*(u.shape[0]-0))
            B = diags((a,b,a), (-1,0,1), format='csr')

        if BC_type == 'neumann':
            # Construct the 2 tri-diags needed for the CN scheme
            a = np.array([-(lmbda/2)]*(u.shape[0]-1))
            b = np.array([1+lmbda]*(u.shape[0]))
            a[-1] = -lmbda
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

            a = np.array([lmbda/2]*(u.shape[0]-1))
            b = np.array([1-lmbda]*(u.shape[0]-0))
            a[-1] = lmbda
            B = diags((a,b,np.flip(a)), (-1,0,1), format='csr')

        if BC_type == 'periodic':
            # Construct the 2 tri-diags needed for the CN scheme
            a = np.array([-(lmbda/2)]*(u.shape[0]-2))
            b = np.array([1+lmbda]*(u.shape[0]-1))
            A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')
            A[-1,0] = -lmbda/2
            A[0,-1] = -lmbda/2

            a = np.array([lmbda/2]*(u.shape[0]-2))
            b = np.array([1-lmbda]*(u.shape[0]-1))
            B = diags((a,b,a), (-1,0,1), format='csr')
            B[-1,0] = lmbda/2
            B[0,-1] = lmbda/2

        return A, B


def solve_pde(L, T, mx, mt, kappa, BC_type, BC, IC, solver):
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
        The type of boundary conditions, either 'dirichlet', 'neumann' or 'periodic'.

    BC : function
        The boundary condition as a callable function. Must take 2 arguments, x and t.
        Can be any function when using BCs of type 'periodic'.

    IC : function
        The initial condition as a callable function. Must take 2 arguments, x and L.
    
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
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

    # initialise the solution matrix
    u = np.zeros((x.size, t.size))

    # Check BC_type
    BC_types = ['neumann', 'dirichlet', 'periodic']
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
        if lmbda >0.5:
            raise ValueError('Lambda greater than 0.5! Consider reducing mx.')
        if lmbda < 0:
            raise ValueError('Lambda less than 0! Wrong args.')
        solver = forward_euler_step
        A = get_matrix(lmbda, BC_type, u, solver)
    elif solver == 'beuler':
        solver = backward_euler_step
        A = get_matrix(lmbda, BC_type, u, solver)
    elif solver == 'cn':
        solver = crank_nicholson_step
        A, B = get_matrix(lmbda, BC_type, u, solver)
    else:
        raise ValueError('Solver specified does not exist!')

    # Get initial conditions and apply
    for i in range(len(x)):
        u[i,0] = IC(x[i], L)

    # Apply boundary condition for t=0
    if not BC_type == 'periodic':
        u[0,0] = BC(0, 0)
        u[-1,0] = BC(L, 0)
    else:
        u = u[0:-1,:]

    for j in range(0, mt):
        # Carry out solver step, including the boundaries
        if not solver == crank_nicholson_step:
            solver(u, A, t, L, BC, BC_type, lmbda, j)
        else:
            solver(u, A, B, t, L, T, BC, BC_type, lmbda, j)
    
    # Add BCs for t=T (necessary for feuler)
    if BC_type == 'dirichlet':
        u[0,-1] = BC(0,T)
        u[-1,-1] = BC(L,T)

    return u, t



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

    # Get numerical solution
    u,t = solve_pde(L, T, mx, mt, kappa, 'dirichlet', homo_BC, u_I, solver='feuler')

    # Plot solution in space and time
    from plots import plot_pde_space_time_solution
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map')

    # Plot x at time T from the exact solution and numerical method.
    xx = np.linspace(0,L,mx+1)
    from plots import plot_pde_specific_time
    plot_pde_specific_time(u, t, 0.3, L, 'Diffusion Solution', u_exact(xx, 0.3, kappa, L))


    u,t = solve_pde(L, T, mx, mt, kappa, 'neumann', non_homo_BC, u_I, solver='feuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map non-homo f')

    u,t = solve_pde(L, T, mx, mt, kappa, 'dirichlet', non_homo_BC, u_I, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map diri non-homo cn')


    u,t = solve_pde(L, T, mx, mt, kappa, 'neumann', homo_BC, u_I, solver='feuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map nonhomo_neu f')
    plot_pde_specific_time(u, t, 0.5, L, 'Specific Time Feuler Neumann Homo', u_exact_nhomo(xx,0.5,kappa,L=1))

    u,t = solve_pde(L, T, mx, mt, kappa, 'periodic', homo_BC, u_I2, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map Periodic cn')

    u,t = solve_pde(L, T, mx, mt, kappa, 'neumann', homo_BC, u_I, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map nonhomo_neu cn')


    return 0

if __name__ == '__main__':
    main()