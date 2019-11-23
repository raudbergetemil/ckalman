#!/usr/bin/python3
#%%
# File for testing functions 
import matplotlib.pyplot as plt
import numpy as np
from state_estimator import *
import scipy.io


def create_trajectory(n,N,Q,f):
    x0 = np.zeros(n)
    x0 = np.random.multivariate_normal(x0, Q)
    x = np.zeros((n,N))

    for i in range(N-1):
        x[:,i+1] = f(x[:,i])
    return x

def create_measurements(x,h):
    N = x.shape[1]
    m = h(x[:,0]).shape[0]
    y = np.zeros((m,N))

    for i in range(N):
        y[:,i] = h(x[:,i])
    return y

def test(pred_fcn, update_fcn):

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
    def real_f(x):
        assert x.shape[0] == n, 'Number of rows in x does not match n'

    
        return A@x + np.random.multivariate_normal(np.zeros((n,)),Q)

    """
    Measurement model y = h(x)
    """
    def real_h(x):
        assert x.shape[0] == n, 'Number of rows in x does not match n'

        return H@x + np.random.multivariate_normal(np.zeros((m,)),R)


    x = create_trajectory(n,N,Q,real_f)
    y = create_measurements(x,real_h)
    x_hat = np.zeros((n,N))
    P_hat = np.zeros((N,n,n))
    P_hat[0,:,:] = Q
    for i in range(N-1):
        x_hat[:,i], P_hat[i,:,:] = update_fcn(x_hat[:,i], P_hat[i,:,:], y[:,i], real_h, R)
        x_hat[:,i+1], P_hat[i+1,:,:] = pred_fcn(x_hat[:,i], P_hat[i,:,:], real_f, Q)
    fig, axis = plt.subplots()
    axis.plot(x_hat[0,:])
    axis.plot(x[0,:])
    plt.show()

pred_fcn = ckf_prediction
update_fcn = ckf_update

#test(pred_fcn, update_fcn)
data = scipy.io.loadmat('test.mat')
print(data)