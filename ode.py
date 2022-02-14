def f(x,t, params):

    return x


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