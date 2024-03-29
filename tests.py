import unittest
import numpy as np
from ode_solvers import solve_ode
from numerical_shooting import numerical_shooting
from ode import *
from pde_solvers import *
from math import pi


class TestODESolvers(unittest.TestCase):

    def test_method_euler(self):
        # Set up ODE problem
        t = np.linspace(0,1,101)
        X0= [1]
        # Solve numerically and get analytic solution
        X = solve_ode('euler', f, t, X0, h_max=0.0001)
        X_true = np.e**(1)

        # Compare numerical and analytic solutions within some tolerance
        self.assertTrue(np.abs(X[-1]-X_true) < 10**-3)

    def test_method_midpoint(self):
        # Set up ODE problem
        t = np.linspace(0,1,101)
        X0= [1]
        # Solve numerically and get analytic solution
        X = solve_ode('midpoint', f, t, X0, h_max=0.001)
        X_true = np.e**(1)

        # Compare numerical and analytic solutions within some tolerance
        self.assertTrue(np.abs(X[-1]-X_true) < 10**-4)

    def test_method_heun3(self):
        # Set up ODE problem
        t = np.linspace(0,1,101)
        X0= [1]
        # Solve numerically and get analytic solution
        X = solve_ode('heun3', f, t, X0, h_max=0.001)
        X_true = np.e**(1)

        # Compare numerical and analytic solutions within some tolerance
        self.assertTrue(np.abs(X[-1]-X_true) < 10**-8)

    def test_method_rk4(self):
        # Set up ODE problem
        t = np.linspace(0,1,101)
        X0= [1]
        # Solve numerically and get analytic solution
        X = solve_ode('rk4', f, t, X0, h_max=0.001)
        X_true = np.e**(1)

        # Compare numerical and analytic solutions within some tolerance
        self.assertTrue(np.abs(X[-1]-X_true) < 10**-8)

    def test_dim2_ODE(self):
        # Construct and solve ODE problem
        t = np.linspace(0,1,101)
        X0 = [0,1]
        X = solve_ode('rk4', g, t, X0)

        # Get analytical solution
        x0_true = np.sin(1)
        x1_true = np.cos(1)
        X_true = np.array([x0_true, x1_true]).transpose()
        # Calculate error
        error = np.abs(X[-1] - X_true)
        # Compare numerical and analytic solutions within some tolerance
        self.assertTrue(error[0] < 10**-8 and error[1] < 10**-8)

    def test_wrong_dims(self):
        test_bool= False
        t = np.linspace(0,1,101)
        X0 = [0]
        # Checks if solve_ode handles wrong dimensions gracefully
        try:
            X = solve_ode('rk4', g, t, X0)
        except ValueError:
            test_bool=True
        assert(test_bool)

class TestNumericalShooting(unittest.TestCase):
    
    def test_dim2_hopf(self):
        # Init params
        sigma = -1
        beta = 1
        
        # Construct phase condition
        def pc_hopf(X0, **params):
            return hopf(X0, 0, params)[0]

        # Carry out shooting and assess result
        X0, T = numerical_shooting([1,1], 5, hopf, pc_hopf, beta=beta, sigma=sigma)
        X = [np.sqrt(beta)*np.cos(T), np.sqrt(beta)*np.sin(T)]

        # Calc error and test
        error = np.abs(X0 - X)
        self.assertTrue(error[0] < 10**-5 and error[1] < 10**-5)

    def test_not_converging(self):

        test_bool = False
        def pc_predator_prey(X0, **params):
            # returns dx/dt at t=0
            return predator_prey(X0, 0, params)[0]
        # Tests gracefulness of non-convergence
        X0= numerical_shooting([1.3, -1.3], 100, predator_prey, pc_predator_prey, a=1, b=0.2, d=0.1)
        assert(X0==[])

class TestPDESolvers(unittest.TestCase):

    def setUp(self):

        def u_exact_dhomo(x,t, kappa, L):
            # the exact solution for homogeneous Dirichlet case
            y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
            return y

        def u_exact_dnonhomo(x,t,kappa,L):
            # The exact solution for non homogeneous Dirichlet case
            y = -1/(np.sin(np.sqrt(2/kappa)*L)) * np.sin(np.sqrt(2/kappa)*(x-L)) * np.e**(-2*t)
            return y

        def u_exact_nhomo(x,t,kappa,L=1):
            # Exact solution for homogeneous Neumann case (noly for L=1)
            y = (2/np.pi) - (4/(3*pi))*np.cos(2*np.pi*x)*(np.e**(-4*(np.pi**2) * kappa * t)) - (4/(15*pi))*np.cos(4*np.pi*x)*(np.e**(-16*(np.pi**2) * kappa * t))
            return y
        
        def u_exact_nonlinear(xx, t_plot, kappa=1, beta = 1):
            # Exact solution for non-linear problem
            sum = 0
            for n in range(100):
                n+=1
                sum += ((((-1)**n) - 1)/n) * np.e**(-t_plot*(((n**2) * (np.pi**2)) - 0.1)) * np.sin(n*np.pi*xx)
            sol = (-2/np.pi)*sum
            return sol

        # Set problem parameters/functions
        self.kappa = 0.1   # diffusion constant
        self.L=1      # length of spatial domain
        self.T=0.5       # total time to solve for

        # Set numerical parameters
        self.mx = 100     # number of gridpoints in space
        self.mt = 1000   # number of gridpoints in time

        # Get exact solution for homogeneous Dirichlet case
        xx = np.linspace(0,self.L,self.mx+1)
        self.exact_sol_dhomo = u_exact_dhomo(xx, self.T, self.kappa, self.L)
        self.exact_sol_dnonhomo = u_exact_dnonhomo(xx, self.T, self.kappa, self.L)
        self.exact_sol_nhomo = u_exact_nhomo(xx, self.T, self.kappa, self.L)
        self.exact_sol_nonlinear = u_exact_nonlinear(xx, self.T)

    def u_I(self, x, L):
        # initial temperature distribution
        y = (np.sin(pi*x/L))
        return y

    def u_I2(self, x, L):
        if x==0:
            return 1
        else:
            return 0

    def homo_BC(self, x, t):
        # Homogeneous BCs, always return 0
        return 0

    def non_homo_BC(self,x,t):
        if x==0:
            return 1*np.e**(-2*t)
        else:
            return 0
        
    def homo_RHS(self,x,t):
        return 0

    ######################################################################################################################
    def test_method_forward_euler_dhomo(self):
        # Test for feuler method w/ homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.homo_BC, self.u_I, solver='feuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dhomo).all() < 10**-4)

    def test_method_backward_euler_dhomo(self):
        # Test for beuler method w/ homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.homo_BC, self.u_I, solver='beuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dhomo).all() < 10**-4)

    def test_method_crank_nicholson_dhomo(self):
        # Test for CN method w/ homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.homo_BC, self.u_I, solver='cn')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dhomo).all() < 10**-4)
        
    ######################################################################################################################

    def test_method_forward_euler_dnonhomo(self):
        # Test for feuler method w/ non-homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.non_homo_BC, self.u_I2, solver='feuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dnonhomo).all() < 10**-4)

    def test_method_backward_euler_dnonhomo(self):
        # Test for beuler method w/ non-homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.non_homo_BC, self.u_I2, solver='beuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dnonhomo).all() < 10**-4)

    def test_method_crank_nicholson_dnonhomo(self):
        # Test for CN method w/ non-homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.non_homo_BC, self.u_I2, solver='cn')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dnonhomo).all() < 10**-4)

    ######################################################################################################################

    def test_method_forward_euler_nhomo(self):
        # Test for feuler method w/ homogeneous neumann BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'neumann', self.homo_BC, self.u_I,solver='feuler')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    def test_method_backward_euler_nhomo(self):
        # Test for beuler method w/ homogeneous neumann BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'neumann', self.homo_BC, self.u_I,solver='beuler')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    def test_method_crank_nicholson_nhomo(self):
        # Test for cn method w/ homogeneous neumann BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'neumann', self.homo_BC, self.u_I,solver='cn')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())
        
    ######################################################################################################################

    def test_method_crank_nicholson_nonlinear(self):
        # Test for CN and non-linear RHS
        def RHS_f(u, x, t):
            return u
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.homo_BC, lambda x,L:1, solver='cn', RHS = RHS_f, kappa = lambda x:np.ones(len(x)), linearity='nonlinear')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nonlinear), [10**-1]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    def test_method_beuler_nonlinear(self):
        # Test for beuler and non-linear RHS
        def RHS_f(u, x, t):
            return u
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', self.homo_BC, lambda x,L:1, solver='beuler', RHS = RHS_f, kappa = lambda x:np.ones(len(x)), linearity='nonlinear')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nonlinear), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    ######################################################################################################################

    def test_robin_BC(self):
        def robin_BC(x, t):
            '''
            Function containing the Robin boundary condition.
            When using Robin boundary conditions the BC function must return alpha, the effect of the dirichlet
            condition as well as gamma, the total effect at the boundary.
            '''
            # Effect of dirichlet
            alpha = lambda t : 1
            # Effect of neumann
            beta = lambda t : 1

            # The dirichlet and neumann effects on the boundary
            dirichlet_u = lambda x,t : 0
            neumann_u = lambda x,t : 0

            # Combination of all effects
            gamma = alpha(t)*dirichlet_u(x,t) + beta(t)*neumann_u(x,t)

            return gamma, alpha(t)

        # Call the pde solver function, using the Crank-Nicholson method
        u, t = solve_pde(self.L, self.T, self.mx, self.mt, 'robin', robin_BC, self.u_I, solver='feuler', kappa = lambda x:x/(x*10))
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    ######################################################################################################################

    def test_periodic_boundaries(self):

        def homo_BC(x,t):
            '''
            Function containing the homogeneous boundary conditions.
            '''
            return 0

        def IC(x, L):
            '''
            The initial conditions for the PDE
            '''
            return x+2

        def exact_sol(xx, t_plot):
            # Exact solution
            sol = 5/2
            sum=0
            for n in range(100):
                n+= 1
                sum += (np.sin(2*n*np.pi*xx)*(np.e**((-4*(np.pi**2) * (n**2) * t_plot))))/(n*np.pi)
            sol = sol - sum
            return sol

        u_exact = exact_sol(np.linspace(0,self.L,self.mx), 0.01)

        u,t = solve_pde(self.L, 0.1, self.mx, self.mt, 'periodic', homo_BC, IC, solver='beuler', kappa = lambda x:np.ones(len(x)))

        bool = np.less(np.abs(u[:,-1] - u_exact), [10]*len(u_exact))
        self.assertTrue(bool.all())

    ######################################################################################################################

    def test_linear_rhs(self):

        def exact_sol(xx, t_plot):
            # Exact solution for linear problem
            sum = 0
            for n in range(100):
                n+=1
                sum += ((np.sin(n*np.pi*xx)*np.e**(-1*(np.pi**2)*(n**2)*t_plot))*(((-1)**n)*(n**2)*(np.pi**2) -  (n**2)*(np.pi**2) - ((-1)**n)))/((n**3)*(np.pi**3))
            sol = -2*sum - ((xx**3)/(6)) + ((xx)/(6))
            return sol

        u_exact = exact_sol(np.linspace(0,self.L,self.mx+1), self.T)

        u,t = solve_pde(self.L, self.T, self.mx, self.mt, 'dirichlet', lambda x,t:0, lambda x,L : 1, solver='cn', 
                RHS = lambda x,t : x, kappa = lambda x:np.ones(len(x)), linearity='linear')

        bool = np.less(np.abs(u[:,-1] - u_exact), [10**-3]*len(u_exact))
        self.assertTrue(bool.all())

if __name__ == '__main__':
    unittest.main()
