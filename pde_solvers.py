import numpy as np
import matplotlib.pyplot as plt
from math import pi

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



def forward_euler_pde(L, T, mx, mt, kappa):
    '''
    
    
    '''

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)

    u = np.zeros((x.size, t.size))

    # Get initial conditions
    for i in range(len(x)):
        u[i,0] = u_I(x[i], L)

    # Apply boundary condition for t=0
    u[0,0] = 0
    u[-1,0] = 0

    # Construct tri-diagonal matrix using lambda
    a = np.array([lmbda]*(x.size-1))
    b = np.array([1-2*lmbda]*(x.size))
    A_FE = tridiag(a,b,a,-1,0,1)

    print(A_FE)

    for j in range(0, mt):
        u[:,j+1] = np.matmul(u[:,j], A_FE)
        
        # Apply boundary conditions
        u[0,j+1]=0
        u[-1,j+1]=0

    return u, t



def main():


    # Set problem parameters/functions
    kappa = 1.0   # diffusion constant
    L=1      # length of spatial domain
    T=0.5       # total time to solve for

    # Set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    u,t = forward_euler_pde(L, T, mx, mt, kappa)

    plt.imshow(u, aspect='auto')
    plt.show()

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