import numpy as np

class Simulator:
    'Simulator object containing functions to create and interact with a simulation'
    def __init__(self, n=1, m=0, o=1, Ts=0.01):

        self.m = m # number of inputs
        self.n = n # number of states
        self.o = o # number of measurements
        
        self.x = np.ones(self.n) # current (hidden) state
        self.y = np.zeros(self.m) # current observation

        self.Ts = 0.01 # sampling time

    
    def process_model(self, x, u): 
        """
        Maps current state to next state

        TODO: 
        - Add noise
        - Create more interesting process models
        """
        
        A = 0.5*np.eye(self.n)
        B = np.eye(self.n, self.m)
        Q = np.eye(self.n, self.n)
        return A@x + B@u + np.random.multivariate_normal(np.zeros((n,)),Q)

    def measurement_model(self, x):
        """
        Maps current state to an observation of the current state
        
        TODO:
        - Add noise
        - More interesting measurement models
        """

        H = np.eye(self.o, self.n)
        R = np.eye(self.o, self.o)
        return H@x + np.random.multivariate_normal(np.zeros((m,)),R)

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
