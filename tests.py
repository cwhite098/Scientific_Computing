import unittest
import numpy as np
from ode_solvers import solve_ode
from numerical_shooting import numerical_shooting
from ode import *

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
        t = np.linspace(0,T,100)
        X = [np.sqrt(beta)*np.cos(T), np.sqrt(beta)*np.sin(T)]

        # Calc error and test
        error = np.abs(X0 - X)
        self.assertTrue(error[0] < 10**-5 and error[1] < 10**-5)






if __name__ == '__main__':
    unittest.main()
