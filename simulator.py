import numpy as np

class Simulator:
    'Simulator object containing functions to create and interact with a simulation'
    def __init__(self):

        self.m = 1 # number of inputs
        self.n = 1 # number of states
        self.o = 1 # number of measurements
        
        self.x = np.ones(self.n) # current (hidden) state
        self.y = np.zeros(self.m) # current observation

        self.Ts = 0.01 # sampling time

    
    def process_model(self, x): 
        """
        Maps current state to next state

        TODO: 
        - Add noise
        - Create more interesting process models
        """
        
        A = 0.5*np.eye(self.n)
        B = np.eye(self.n, self.m)
        noise = None

        return A@x #+ noise

    def measurement_model(self, x):
        """
        Maps current state to an observation of the current state
        
        TODO:
        - Add noise
        - More interesting measurement models
        """

        H = np.eye(self.o, self.n)

        noise = None
        return H@x #+ noise
    def step(self):
        """
        Increase time in simulation

        TODO: 
        - Implement rewards to train RL algorithms
        """
        self.x = self.process_model(self.x)
        return self.measurement_model(self.x)