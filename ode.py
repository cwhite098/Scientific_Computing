def f(x,t):

    return x


def g(X,t):
    x = X[0]
    y = X[1]
    dxdt = y
    dydt = -x
    X = [dxdt, dydt]

    return X