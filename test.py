#!/usr/bin/python3
#%%
# File for testing functions 
import matplotlib.pyplot as plt
import numpy as np
from state_estimator import *
import scipy.io
from simulator import Simulator
from control import lqr_gain


def create_trajectory(n, N, Q, f, m=1):
    x0 = np.zeros(n)
    x0 = np.random.multivariate_normal(x0, Q)
    x = np.zeros((n,N))
    u = np.zeros((m,N))

    for i in range(N-1):
        x[:,i+1] = f(x[:,i], u[:,i])
    return x, u

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
    o = 2
    Q = np.eye(n)
    R = np.eye(m)
    A = 0.9*np.eye(n)
    B = np.eye(n)
    H = np.eye(o)
    N = 100
    """ 
    Process model x_{k+1} = f(x_{k})
    """
    def real_f(x, u):
        assert x.shape[0] == n, 'Number of rows in x does not match n'

    
        return A@x + B@u + np.random.multivariate_normal(np.zeros((n,)),Q)

    """
    Measurement model y = h(x)
    """
    def real_h(x):
        assert x.shape[0] == n, 'Number of rows in x does not match n'

        return H@x + np.random.multivariate_normal(np.zeros((m,)),R)


    x,u = create_trajectory(n,N,Q,real_f, m)
    y = create_measurements(x,real_h)
    x_hat = np.zeros((n,N))
    P_hat = np.zeros((N,n,n))
    P_hat[0,:,:] = Q

    sim = Simulator(n,m,o)
    u = np.zeros((m,N))

    # Calculate LQR gain
    K = lqr_gain(A,B,Q,R)

    for i in range(N-1):
        # Calculate control signal 
        u[:,i] = -K@x_hat[:,i]
        
        # Take step in simulator environment and observe output
        y[:,i+1] = sim.step(u[:,i])

        # Predict from current estimate
        x_hat[:,i+1], P_hat[i+1,:,:] = pred_fcn(x_hat[:,i], P_hat[i,:,:], A,B, u[:,i], Q)

        # Update estimate 
        x_hat[:,i+1], P_hat[i+1,:,:] = update_fcn(x_hat[:,i+1], P_hat[i+1,:,:], y[:,i+1], H, R)

        # Save real current state
        x[:,i+1] = sim.get_current_state()

    fig, axis = plt.subplots()
    axis.plot(x_hat[0,:])
    axis.plot(x[0,:])
    plt.show()

pred_fcn = linear_kalman_prediction
update_fcn = linear_kalman_update

test(pred_fcn, update_fcn)
#data = scipy.io.loadmat('test.mat')