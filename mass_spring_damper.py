import numpy as np

class MassSpringDamper:
    
    def __init__(self, m, b, k, dt):
        
        self.dt = dt
        self.m = m
        self.b = b
        self.k = k
        
        self.A = np.array([[0,1],[-k/m,-b/m]])
        self.B = np.array([[0],[1/m]])
        
    def calculate_xdot(self, x, u):
        
        return self.A * x + self.B * u
    
    def forward_simulate(self, x, u):
        
        xdot = self.calculate_xdot(x, u)
        xNext = x + self.dt * xdot
        
        return xNext