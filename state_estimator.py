#!/usr/bin/python3
#%%
# import keras
import numpy as np
import matplotlib.pyplot as plt

n = 2
m = 2
Q = np.eye(n)
R = np.eye(m)
A = np.eye(n)
H = np.eye(m)
N = 100


""" 
Process model x_{k+1} = f(x_{k})
"""
def f(x):
    assert x.shape[0] == n, 'Number of rows in x does not match n'

    
    return np.matmul(A,x) + np.random.multivariate_normal(np.zeros((n,)),Q)

"""
Measurement model y = h(x)
"""
def h(x):
    assert x.shape[0] == n, 'Number of rows in x does not match n'

    return np.matmul(H,x) + np.random.multivariate_normal(np.zeros((m,)),R)


def create_trajectory():
    x0 = np.zeros(n)
    x0 = np.random.multivariate_normal(x0, Q)
    x = np.zeros((n,N))

    for i in range(N-1):
        x[:,i+1] = f(x[:,i])
    return x

def create_measurements(x):
    y = np.zeros((m,N))

    for i in range(N):
        y[:,i] = h(x[:,i])
    return y

def kalman_prediction(x, P):
    return (f(x), A@P@A + Q)

def kalman_update(x, P, y):
    S = H@P@H.T + R
    K = P@H.T@np.linalg.inv(S)
    return (x + K@(y-f(x)), P - K@H@P)

# def create_network():
    # model = keras.Sequential()
    # model.add(keras.layers.LSTM(n,activation=None))

def test():
    x = create_trajectory()
    y = create_measurements(x)
    x_hat = np.zeros((n,N))
    P_hat = np.zeros((N,n,n))
    for i in range(N-1):
        x_hat[:,i], P_hat[i,:,:] = kalman_update(x_hat[:,i], P_hat[i,:,:], y[:,i])
        x_hat[:,i+1], P_hat[i+1,:,:] = kalman_prediction(x_hat[:,i], P_hat[i,:,:])
    fig, axis = plt.subplots()
    axis.plot(x_hat[0,:])
    axis.plot(x[0,:])


test()


#%%
