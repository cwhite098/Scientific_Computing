def f(x,t, params):

    return x

def f2(x,t, params):
    a = params['a']
    return x*a


def g(X,t, params):
    
    x = X[0]
    y = X[1]
    dxdt = y
    dydt = -x
    X = [dxdt, dydt]

    return X

def predator_prey(X, t, params):

    a = params['a']
    b = params['b']
    d = params['d']

    x = X[0]
    y = X[1]

    dxdt = x*(1-x) - (a*x*y) / (d+x)
    dydt = b*y*(1-(y/x))

    X = [dxdt, dydt]

    return X


def hopf(X, t, params):

    beta = params['beta']
    sigma = params['sigma']

    u1 = X[0]
    u2 = X[1]

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    X = [du1dt, du2dt]

    return X


def hopf3D(X, t, params):

    beta = params['beta']
    sigma = params['sigma']

    u1 = X[0]
    u2 = X[1]
    u3 = X[2]

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    du3dt = -u3

    X = [du1dt, du2dt, du3dt]

    return X