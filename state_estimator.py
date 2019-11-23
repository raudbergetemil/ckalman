#!/usr/bin/python3

import numpy as np

def linear_kalman_prediction(x, P, A, B, u, Q):
    """
    Prediction step of linear Kalman filter
    """

    return (A@x + B@u, A@P@A.T + Q)

def linear_kalman_update(x, P, y, H, R):
    """
    Update step of linear Kalman filter
    """

    S = H@P@H.T + R
    K = P@H.T@np.linalg.inv(S)
    return (x + K@(y-H@x), P - K@H@P)

def ekf_prediction(x, P, f, dfdx, Q):
    """
    Prediction step of EKF
    """

    return (f(x), dfdx@P@dfdx.T + Q)

def ekf_update(x, P, y, h, dhdx, R):
    """
    Update step of EKF
    """
    
    S = dhdx@P@dhdx.T + R
    K = P@dhdx.T@np.linalg.inv(S)
    return (x + K@(y-h(x)), P - K@dhdx@P)

def sigma_points(x, P, ftype):
    """ 
    Returns the sigma points given x, P and the filter type
    """
    n = x.shape[0]
    cholP = np.linalg.cholesky(P)
    sqrtn = np.sqrt(n)

    if ftype == 'ckf':
        sp = np.zeros((n, 2*n))
        for i in range(n):
            sp[:,i] = x + sqrtn*cholP[:,i]
            sp[:,i+n] = x - sqrtn*cholP[:,i]
            
    elif ftype == 'ukf':
        sp = np.zeros(n, 2*n + 1)
        sp[:,i] = x

        for i in range(1, n):
            sp[:,i] = x + sqrtn*cholP[:,i]
            sp[:,i+n] = x - sqrtn*cholP[:,i]
    return sp

def ckf_prediction(x, P, f, u, Q):
    """
    Prediction step of CKF
    TODO: Make sure P stays pos def
    """

    n = x.shape[0] 
    w = 1/(2*n)
    sp = sigma_points(x, P, 'ckf')
    x_new = np.zeros(n)
    P_new = np.zeros((n,n))

    for i in range(2*n):
        x_new += f(sp[:,i], u)*w

    for i in range(2*n):
        P_new += f(sp[:,i]-x_new, u)*(f(sp[:,i], u)-x_new).T*w
    P_new += Q

    # Make sure P is pos def
    if min(np.linalg.eigvalsh(P_new)) <= 0:
        e, v = np.linalg.eigh(P_new)
        e[e < 0] = 1e-1
        P_new = v@np.diag(e)@np.linalg.inv(v)
    
    if min(np.linalg.eigvalsh(P_new)) <= 0:
        print('')
    return x_new, P_new

def ckf_update(x, P, y, h, R):
    """
    Update step of CKF
    TODO: Make sure P stays pos def
    """

    n = x.shape[0]
    m = y.shape[0]
    w = 1/(2*n)
    sp = sigma_points(x, P, 'ckf')
    P_xy = np.zeros((n,n))
    S = np.zeros((n,n))
    y_hat = np.zeros((m,))

    for i in range(2*n):
        y_hat += h(sp[:,i])*w

    for i in range(2*n):
        P_xy += (sp[:,i]-x)*(h(sp[:,i])-y_hat).T*w
        S += ((h(sp[:,i])-y_hat)*h(sp[:,i])-y_hat).T*w
    S += R
    
    K = P_xy.T@np.linalg.inv(S)

    P_new = P - K@S@K.T
    # Make sure P is pos def
    if min(np.linalg.eigvalsh(P_new)) <= 0:
        e, v = np.linalg.eigh(P_new)
        e[e < 0] = 1e-1
        P_new = v@np.diag(e)@np.linalg.inv(v)
    return (x + K@(y-y_hat), P_new)

def ukf_prediction():
    """
    Prediction step of UKF
    TODO: Implement!
    """

    # w0 = 1 - n/3 # noise assumed gaussian
    # w = (1-w0)/(2*n)
    return None

def ukf_update():
    """
    Update step of UKF
    TODO: Implement!
    """
    return None
