from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags, vstack, hstack, csr_matrix
from scipy.sparse.linalg import spsolve

def u_I(x, L):
    # initial temperature distribution
    y = (np.sin(pi*x/L))
    return y

def u_exact(x,t, kappa, L):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    '''
    Function that constructs a tri-diagonal matrix for use in the Forward Euler PDE solver.

    Parameters
    ----------
    a : np.array
        A numpy array containing the values to assign to diagonal k1.

    b : np.array
        A numpy array containing the values to assign to diagonal k2.

    c : np.array
        A numpy array containing the values to assign to diagonal k3.

    k1, k1, k3 : int
        The diagonal to assign array a to. (k1=0 gives leading diagonal, k1=-1 gives the subdiagonal
        and k1=1 gives the superdiagonal)

    Returns
    -------
    A : np.array
        A numpy array containing the constructed matrix.
    '''
    A = np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
    return A



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
        The type of boundary conditions, either 'dirichlet' or 'neumann'.
    
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
        The type of boundary conditions, either 'dirichlet' or 'neumann'.
    
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
        u[0,j] += lmbda * 4 * L/u.shape[0]
        u[-1,j] += lmbda * 4 * L/u.shape[0]
        # Solve
        u[:,j+1] = spsolve(A, u[:,j])

    return u

def crank_nicholson_step(u, A, B, t, L, T, BC, BC_type, lmbda, j):
    '''
    Function that carries out one step of the fcrank-nicholson method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

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
        The type of boundary conditions, either 'dirichlet' or 'neumann'.
    
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

    return u


def get_matrix(lmbda, BC_type, u, solver):

    # Matrices for forward Euler
    if solver == forward_euler_step:

        if BC_type == 'dirichlet':
            a = np.array([lmbda]*(u.shape[0]-3))
            b = np.array([1-2*lmbda]*(u.shape[0]-2))

        if BC_type == 'neumann':
            a = np.array([lmbda]*(u.shape[0]-1))
            b = np.array([1-2*lmbda]*(u.shape[0]))
            a[-1] = a[-1]*2

        # Get diagonal matrix
        A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')
        return A


    if solver == backward_euler_step:

        if BC_type == 'dirichlet':
            a = np.array([-lmbda]*(u.shape[0]-3))
            b = np.array([1+2*lmbda]*(u.shape[0]-2))

        if BC_type == 'neumann':
            a = np.array([-lmbda]*(u.shape[0]-1))
            b = np.array([1+2*lmbda]*(u.shape[0]))
            a[-1] = a[-1]*2

        # Get diagonal matrix
        A = diags((a,b,np.flip(a)), (-1,0,1), format='csr')
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
        The type of boundary conditions, either 'dirichlet' or 'neumann'.

    BC : function
        The boundary condition as a callable function.

    IC : function
        The initial condition as a callable function.
    
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
    u[0,0] = BC(0, 0)
    u[-1,0] = BC(L, 0)

    for j in range(0, mt):

        # Carry out solver step, including the boundaries
        if not solver == crank_nicholson_step:
            solver(u, A, t, L, BC, BC_type, lmbda, j)
        else:
            solver(u, A, B, t, L, T, BC, BC_type, lmbda, j)
    
    # Add BCs for t=T (necessary for feuler)
    u[0,-1] = BC(0,T)
    u[-1,-1] = BC(L,T)

    return u, t



def main():

    # Set problem parameters/functions
    kappa = 0.1   # diffusion constant
    L=2      # length of spatial domain
    T=1      # total time to solve for

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

    # Get numerical solution
    u,t = solve_pde(L, T, mx, mt, kappa, 'dirichlet', non_homo_BC, u_I, solver='feuler')

    # Plot solution in space and time
    from plots import plot_pde_space_time_solution
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map')

    # Plot x at time T from the exact solution and numerical method.
    xx = np.linspace(0,L,mx+1)
    from plots import plot_pde_specific_time
    plot_pde_specific_time(u, t, 0.3, L, 'Diffusion Solution', u_exact(xx, 0.3, kappa, L))


    

    u,t = solve_pde(L, T, mx, mt, kappa, 'neumann', non_homo_BC, u_I, solver='beuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map non-homo f')

    u,t = solve_pde(L, T, mx, mt, kappa, 'dirichlet', non_homo_BC, u_I, solver='cn')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map diri non-homo cn')


    u,t = solve_pde(L, T, mx, mt, kappa, 'neumann', homo_BC, u_I, solver='feuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map homo_neu f')

    u,t = solve_pde(L, T, mx, mt, kappa, 'neumann', non_homo_BC, u_I, solver='feuler')
    plot_pde_space_time_solution(u, L, T, 'Space Time Solution Heat Map nonhomo_neu f')


    return 0

if __name__ == '__main__':
    main()