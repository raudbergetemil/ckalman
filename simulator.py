import numpy as np

class Simulator:
    'Simulator object containing functions to create and interact with a simulation'
    def __init__(self, n=1, m=0, o=1, Ts=0.01):

        self.m = m # number of inputs
        self.n = n # number of states
        self.o = o # number of measurements
        
        self.x = np.ones(self.n) # current (hidden) state
        self.y = np.zeros(self.m) # current observation

        self.Ts = 0.1 # sampling time
    
    def RK4_next_step(self,x,u,f):
        """
        Returns the next state based on the Explicit Runge Kutta 4 method. 
        Assumes that control signal is piecewise constant. Does not handle explicit 
        time dependency.
        """

        # Calculate coefficients
        k1 = self.Ts*f(x,u)
        k2 = self.Ts*f(x + k1/2,u)
        k3 = self.Ts*f(x + k2/2, u)
        k4 = self.Ts*f(x + k3)

        # Calculate and return value at next timestep
        return x + 1/6*(k1+k2+k3+k4)
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
        # Discrete predator-prey model
        # https://www.researchgate.net/publication/335681339_Dynamics_of_a_discrete_nonlinear_prey-predator_model
        # new_x = np.array([0.2*x[0]*(1-x[0]) - 0.43*x[0]*x[1] + u[0], 3*x[0]*x[1] + u[1]])

        # Nonlinear model
        # Euler forward discretized Volterra lotka
        alpha = 0.5
        beta = 0.2
        delta = 0.6
        gamma = 0.3
        x1 = x[0] + self.Ts*(alpha*x[0] - beta*x[0]*x[1] + u[0])
        x2 = x[1] + self.Ts*(delta*x[0]*x[1] - gamma*x[1] + u[1])
        return np.array([x1,x2])

    def measurement_model(self, x):
        """
        Maps current state to an observation of the current state
        
        TODO:
        - Add noise
        - More interesting measurement models
        """

        H = np.eye(self.o, self.n)
        R = 0.0001*np.eye(self.o, self.o)
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
