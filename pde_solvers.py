from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags
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



def forward_euler_step(u, A, j):
    '''
    Function that carries out one step of the forward Euler numerical method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

    A : scipy sparse matrix
        A scipy sparse matrix that is used to calculate the next time step of the solution.

    j : int
        The time at which the forward Euler step is applied.

    Returns
    -------
    u : np.array
        The updated solution matrix.
    '''
    u[:,j+1] = A.dot(u[:,j])

    return u

def backward_euler_step(u, A, j):
    '''
    Function that carries out one step of the backward Euler numerical method for approximating
    PDEs.

    Parameters
    ----------
    u : np.array
        A numpy array containing the solution of the PDE, excluding the boundaries.

    A : scipy sparse matrix
        A scipy sparse matrix that is used to calculate the next time step of the solution.

    j : int
        The time at which the forward Euler step is applied.

    Returns
    -------
    u : np.array
        The updated solution matrix.
    '''

    u[:,j+1] = spsolve(A, u[:,j])

    return u

def crank_nicholson_step(u, A, B, j):
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

    j : int
        The time at which the forward Euler step is applied.

    Returns
    -------
    u : np.array
        The updated solution matrix.
    '''

    u[:,j+1] = spsolve(A, B.dot(u[:,j]))

    return u



def solve_pde(L, T, mx, mt, kappa, solver):
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

    if solver == 'feuler':
        # Checks if solver will be stable with this lambda value
        if lmbda >0.5:
            raise ValueError('Lambda greater than 0.5! Consider reducing mx.')
        if lmbda < 0:
            raise ValueError('Lambda less than 0! Wrong args.')
        solver = forward_euler_step

        # Construct tri-diagonal matrix using lambda
        a = np.array([lmbda]*(x.size-3))
        b = np.array([1-2*lmbda]*(x.size-2))
        A = diags((a,b,a), (-1,0,1), format='csr')

    elif solver == 'beuler':
        solver = backward_euler_step
        # Construct tri-diag matrix for the numerical scheme
        a = np.array([-lmbda]*(x.size-3))
        b = np.array([1+2*lmbda]*(x.size-2))
        A = diags((a,b,a), (-1,0,1), format='csr')

    elif solver == 'cn':
        solver = crank_nicholson_step
        # Construct the 2 tri-diags needed for the CN scheme
        a = np.array([-(lmbda/2)]*(x.size-3))
        b = np.array([1+lmbda]*(x.size-2))
        A = diags((a,b,a), (-1,0,1), format='csr')

        a = np.array([lmbda/2]*(x.size-3))
        b = np.array([1-lmbda]*(x.size-2))
        B = diags((a,b,a), (-1,0,1), format='csr')

    else:
        raise ValueError('Solver specified does not exist!')

    # initialise the solution matrix
    u = np.zeros((x.size, t.size))

    # Get initial conditions
    for i in range(len(x)):
        u[i,0] = u_I(x[i], L)

    # Apply boundary condition for t=0
    u[0,0] = 0
    u[-1,0] = 0

    for j in range(0, mt):

        # Apply boundary conditions
        u[0,j+1]=0
        u[-1,j+1]=0
        
        # Carry out solver step, excluding the boundaries
        if not solver == crank_nicholson_step:
            solver(u[1:-1], A, j)
        else:
            solver(u[1:-1], A, B, j)
    
    return u, t



def main():

    # Set problem parameters/functions
    kappa = 0.1   # diffusion constant
    L=1      # length of spatial domain
    T=0.5       # total time to solve for

    # Set numerical parameters
    mx = 100     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    # Get numerical solution
    u,t = solve_pde(L, T, mx, mt, kappa, solver='feuler')

    # Plot solution in space and time
    plt.imshow(u, aspect='auto')
    plt.show()

    # Plot x at time T from the exact solution and numerical method.
    xx = np.linspace(0,L,mx+1)
    exact_sol = u_exact(xx, T, kappa, L)
    print(exact_sol)
    plt.plot(xx, u[:,-1], 'o', c='r')
    plt.plot(xx, exact_sol, 'b-', label='exact')
    plt.legend()
    plt.show()




    return 0

if __name__ == '__main__':
    main()