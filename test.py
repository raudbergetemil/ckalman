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
    R = 0.0001*np.eye(m)
    A = 0.5*np.eye(n)
    B = np.eye(n)
    H = np.eye(o)
    N = 1000
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

    def f(x,u):

        # return A@x + B@u
        #alpha = 0.5
        #beta = 0.2
        #delta = 0.6
        #gamma = 0.3
        #x1 = x[0] + 0.1*(alpha*x[0] - beta*x[0]*x[1] + u[0])
        #x2 = x[1] + 0.1*(delta*x[0]*x[1] - gamma*x[1] + u[1])
        #return np.array([x1,x2])

        # Discretized volterra lotka with mu = 1
        Ts = 0.1
        mu = 1
        x1_dot = mu*(x[0] - 1/3*x[0]**3 - x[1]) + u[0]
        x2_dot = 1/mu*x[0] 
        dxdt = np.array([x1_dot, x2_dot])

        return x + Ts*dxdt


    def h(x):
        return H@x 
    
    def fl_u(x):
        """
        Calculates the feedback linearizing control signal given state x
        """
        # a = 0.01
        # u1 =  - x[0]*(0.2*(1-x[0])-0.43*x[1])
        # u2 =  - 3*x[0]*x[1]
        # return np.array([u1,u2])

        # For predator-prey model in 'simulator.py'
        # alpha = 0.5
        # beta = 0.2
        # delta = 0.6
        # u1 = -x[0]*(1+alpha) +  beta*x[0]*x[1]
        # u2 = -delta*x[0]*x[1] 
        # return np.array([u1,u2])

        # For Van der pol oscillator
        Ts = 0.1
        mu = 1
        x1_dot = mu*(x[0] - 1/3*x[0]**3 - x[1]) 
        x2_dot = 1/mu*x[0] 

        u1 = -(x[0] + Ts*x1_dot) - 0.1*x[0]
        u2 = 0
        return np.array([u1,u2])

    x = np.zeros((n,N))
    y = np.zeros((o,N))
    x_hat = np.zeros((n,N))
    P_hat = np.zeros((N,n,n))
    P_hat[0,:,:] = 10*Q

    sim = Simulator(n,m,o)
    u = np.zeros((m,N))

    # Calculate LQR gain
    K = lqr_gain(A,B,Q,R)
    fig, axis = plt.subplots()

    for i in range(N-1):
        # Save real current state
        x[:,i] = sim.get_current_state()

        # Calculate control signal 
        u[:,i] = fl_u(x_hat[:,i])
        
        # Take step in simulator environment and observe output
        y[:,i+1] = sim.step(u[:,i])

        # x_hat[:,i+1] = y[:,i+1]
        # Predict from current estimate
        x_hat[:,i+1], P_hat[i+1,:,:] = pred_fcn(x_hat[:,i], P_hat[i,:,:], f, u[:,i], Q)
       
        # Update estimate 
        x_hat[:,i+1], P_hat[i+1,:,:] = update_fcn(x_hat[:,i+1], P_hat[i+1,:,:], y[:,i+1], h, R)
        

    axis.plot(x_hat[0,:])
    axis.plot(x[0,:])
    #axis.plot(y[0,:], 'b.', markersize=0.5)
    axis.plot(u[0,:])
    axis.legend(['Estimate 1', 'Real 1', 'Measurement'])
    plt.show()

pred_fcn = ckf_prediction
update_fcn = ckf_update

test(pred_fcn, update_fcn)
#data = scipy.io.loadmat('test.mat')