#!/usr/bin/python3

import numpy as np

"""
Prediction step of linear Kalman filter
"""
def linear_kalman_prediction(x, P, A, Q):
    return (A@x, A@P@A.T + Q)

"""
Update step of linear Kalman filter
"""
def linear_kalman_update(x, P, y, H, R):
    S = H@P@H.T + R
    K = P@H.T@np.linalg.inv(S)
    return (x + K@(y-H@x), P - K@H@P)

"""
Prediction step of EKF
"""
def ekf_prediction(x, P, f, dfdx, Q):
    return (f(x), dfdx@P@dfdx.T + Q)

"""
Update step of EKF
"""
def ekf_update(x, P, y, h, dhdx, R):

    S = dhdx@P@dhdx.T + R
    K = P@dhdx.T@np.linalg.inv(S)
    return (x + K@(y-h(x)), P - K@dhdx@P)

""" 
Returns the sigma points given x, P and the filter type
"""
def sigma_points(x, P, ftype):
    
    n = x.shape[0]
    cholP = np.linalg.cholesky(P)

    if ftype == 'ckf':
        sp = np.zeros(n, 2*n)
        sqrtn = np.sqrt(n)
        for i in range(n):
            sp[:,i] = x + sqrtn*cholP[:,i]
            sp[:,i+n] = x - sqrtn*cholP[:,i]
            
    elif ftype == 'ukf':
        sp = np.zeros(n, 2*n + 1)
        sqrtn = np.sqrt(n)
        sp[:,i] = x

        for i in range(1, n):
            sp[:,i] = x + sqrtn*cholP[:,i]
            sp[:,i+n] = x - sqrtn*cholP[:,i]
    return sp

"""
Prediction step of CKF
TODO: Implement!
"""
def ckf_prediction(x, P, f, Q):
    n = x.shape[0] 
    w = 1/(2*n)
    sp = sigma_points(x, P, 'ckf')
    x_new = np.zeros(n)
    P_new = np.zeros((n,n))

    for i in range(2*n):
        x_new += f(sp[:,i])*w

    for i in range(2*n):
        P_new += f(sp[:,i]-x_new)*(f(sp[:,i])-x_new).T*w
    P_new += Q

    return x_new, P_new

"""
Update step of CKF
TODO: Implement!
"""
def ckf_update(x, P, y, h, R):
    n = x.shape[0] 
    w = 1/(2*n)
    sp = sigma_points(x, P, 'ckf')
    P_xy = np.zeros((n,n))
    S = np.zeros((n,n))

    for i in range(2*n):
        y_hat += h(sp[:,i])*w

    for i in range(2*n):
        P_xy += (sp[:,i]-x)*(h(sp[:,i])-y_hat).T*w
        S += ((h(sp[:,i])-y_hat)*h(sp[:,i])-y_hat).T*w
    S += R
    
    K = P_xy.T@np.linalg.inv(S)
    return (x + K@(y-y_hat), P - K@S@K.T)

"""
Prediction step of UKF
TODO: Implement!
"""
def ukf_prediction():
    # w0 = 1 - n/3 # noise assumed gaussian
    # w = (1-w0)/(2*n)
    return None

"""
Update step of UKF
TODO: Implement!
"""
def ukf_update():
    return None
