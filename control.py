import numpy as np
from scipy.linalg import solve_discrete_are

def lqr_gain(A,B,Q,R):
    "Calculates the LQR gain K"
    P = solve_discrete_are(A,B,Q,R)
    return np.linalg.inv(R + B.T @ P @ B)@(B.T @ P @ A)

