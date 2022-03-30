import unittest
import numpy as np
from ode_solvers import solve_ode
from numerical_shooting import numerical_shooting
from ode import *
from pde_solvers import *
import plots as p


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

    ######################################################################################################################
    def test_method_forward_euler_dhomo(self):
        # Test for feuler method w/ homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'dirichlet', self.homo_BC, self.u_I, solver='feuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dhomo).all() < 10**-4)

    def test_method_backward_euler_dhomo(self):
        # Test for beuler method w/ homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'dirichlet', self.homo_BC, self.u_I, solver='beuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dhomo).all() < 10**-4)

    def test_method_crank_nicholson_dhomo(self):
        # Test for CN method w/ homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'dirichlet', self.homo_BC, self.u_I, solver='cn')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dhomo).all() < 10**-4)
        
    ######################################################################################################################

    def test_method_forward_euler_dnonhomo(self):
        # Test for feuler method w/ non-homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'dirichlet', self.non_homo_BC, self.u_I2, solver='feuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dnonhomo).all() < 10**-4)

    def test_method_backward_euler_dnonhomo(self):
        # Test for beuler method w/ non-homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'dirichlet', self.non_homo_BC, self.u_I2, solver='beuler')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dnonhomo).all() < 10**-4)

    def test_method_crank_nicholson_dnonhomo(self):
        # Test for CN method w/ non-homogeneous Dirichlet BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'dirichlet', self.non_homo_BC, self.u_I2, solver='cn')
        self.assertTrue(np.abs(u[:,-1] - self.exact_sol_dnonhomo).all() < 10**-4)

    ######################################################################################################################

    def test_method_forward_euler_nhomo(self):
        # Test for feuler method w/ homogeneous neumann BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'neumann', self.homo_BC, self.u_I, solver='feuler')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    def test_method_backward_euler_nhomo(self):
        # Test for beuler method w/ homogeneous neumann BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'neumann', self.homo_BC, self.u_I, solver='beuler')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())

    def test_method_crank_nicholson_nhomo(self):
        # Test for cn method w/ homogeneous neumann BCs
        u,t = solve_pde(self.L, self.T, self.mx, self.mt, self.kappa, 'neumann', self.homo_BC, self.u_I, solver='cn')
        bool = np.less(np.abs(u[:,-1] - self.exact_sol_nhomo), [10**-3]*len(self.exact_sol_nhomo))
        self.assertTrue(bool.all())
        
    ######################################################################################################################


if __name__ == '__main__':
    unittest.main()
