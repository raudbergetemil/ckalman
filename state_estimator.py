#!/usr/bin/python3
#%%
# import keras
import numpy as np

def kalman_prediction(x, P, A, Q):
    return (A@x, A@P@A + Q)

def kalman_update(x, P, y, H, R):
    S = H@P@H.T + R
    K = P@H.T@np.linalg.inv(S)
    return (x + K@(y-H@x), P - K@H@P)

