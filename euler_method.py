from ode import f
import numpy as np


def euler_step(X, t, h, f):

    dxdt = f(X,t)
    Xnew = X + h*dxdt
    
    return Xnew


def solve_to(t, t_end, X0, h_max, f, solver):

    h = round(t_end - t, 9)

    if h > h_max:
        while round(t,9) < round(t_end, 9):
            X1 = solver(X0, t, h_max, f)
            
            X0 = X1
            t += h_max


        if (t < t_end) and (t + h_max > t_end):
            X1 = solver(X0, t, t_end-t, f)
            print('finishing off')

    else:
        X1 = solver(X0, t, h, f)

    return X1

def solve_ode(method, f, t, X0, h_max = 0.1):

    X = np.zeros(len(t))
    X[0] = X0

    for i in range(len(t)-1):
        t0 = t[i]
        t1 = t[i+1]
        X[i+1] = solve_to(t0, t1, X[i], h_max, f, method)

    return X



def main():

    X0 = 1
    t = np.linspace(0.0, 1.0, 3)
    X = solve_ode(euler_step, f, t, X0)
    print(t)
    print(X)

    return 0


if __name__ == '__main__':
    main()
