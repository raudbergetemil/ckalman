import numpy as np

class Simulator:
    'Simulator object containing functions to create and interact with a simulation'
    def __init__(self, n=1, m=0, o=1, Ts=0.01):

        self.m = m # number of inputs
        self.n = n # number of states
        self.o = o # number of measurements
        
        self.x = 5*np.ones(self.n) # current (hidden) state
        self.y = np.zeros(self.m) # current observation

        self.Ts = 0.0000001 # sampling time

    
    def process_model(self, x, u): 
        """
        Maps current state to next state

        """
        # # Linear model 
        # A = 0.5*np.eye(self.n)
        # B = np.eye(self.n, self.m)
        
        # Q = np.eye(self.n, self.n)
        # return A@x + B@u + np.random.multivariate_normal(np.zeros((self.n,)),Q)

        # Nonlinear model
        # Discrete Volterra-Lotka predator-prey model
        new_x = np.array([0.1*x[0]*(1-x[0]) - 0.7*x[0]*x[1], 0.1*x[0]*x[1]])
        return new_x

    def measurement_model(self, x):
        """
        Maps current state to an observation of the current state
        
        TODO:
        - Add noise
        - More interesting measurement models
        """

        H = np.eye(self.o, self.n)
        R = np.eye(self.o, self.o)
        return H@x + np.random.multivariate_normal(np.zeros((self.m,)),R)

    def step(self, u):
        """
        Increase time in simulation

        TODO: 
        - Implement rewards to train RL algorithms
        """
        self.x = self.process_model(self.x, u)
        return self.measurement_model(self.x)
    
    def get_current_state(self):
        return self.x
