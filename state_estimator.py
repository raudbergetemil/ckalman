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

