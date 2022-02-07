from ode import f
import numpy as np


def euler_step(X, t, h, f):

    dxdt = f(X,t)
    Xnew = X + h*dxdt
    
    return Xnew


def solve_to(t, t_end, X, h_max, f, solver):


    while t < t_end:
        Xnew = solver(X, t, h_max, f)
        print(Xnew)
        X = Xnew
        t = t + h_max

    if not t == t_end:
        h = t - t_end
        Xnew = solver(X, t, h, f)
        X = Xnew
        t = t + h
        print('h', h)
        print(Xnew)


    return X

def solve_ode(method, f, t, X0):

    X = [X0]
    h_max = 0.1

    for i in range(len(t)-1):
        t0 = t[i]
        t1 = t[i+1]
        Xnew = solve_to(t0, t1, X[i], h_max, f, method)
        X.append(Xnew)

    return X



def main():

    #solve_to(0, 1, 1, 0.001, f, euler_step)
    X0 = 1
    t = np.linspace(0, 1, 10)
    X = solve_ode(euler_step, f, t, X0)
    print(X)

    return 0


if __name__ == '__main__':
    main()
