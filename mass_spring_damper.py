import numpy as np
import matplotlib.pyplot as plt

class MassSpringDamper:
    
    def __init__(self, m, b, k, dt):
        
        self.dt = dt
        
        self.A = np.array([[0,1],[-k/m,-b/m]])
        self.B = np.array([[0],[1/m]])
        
    def calculate_xdot(self, x, u):
        
        return self.A @ x + self.B * u
    
    def forward_simulate(self, x, u):
        
        xdot = self.calculate_xdot(x, u)
        xNext = x + self.dt * xdot
        
        return xNext
    
    def simulate(self, totalTime, x):
        
        numForwardSimulationSteps = int(totalTime / self.dt)
        elapsedTime = 0
        xHist = np.zeros((numForwardSimulationSteps, 2))
        timeHist = np.zeros(numForwardSimulationSteps)
        
        for i in range(0, numForwardSimulationSteps):
            
            u = 10.0
            x = self.forward_simulate(x, u)
                        
            elapsedTime = elapsedTime + self.dt
            
            xHist[i, :] = x.flatten()
            timeHist[i] = elapsedTime
            
        return xHist, timeHist
    
    def plot_hist(self, x, totalTime):
        
        plt.plot(totalTime, x[:, 0])
        plt.show()
            
if __name__ == "__main__":
    
    system = MassSpringDamper(50.0, 10.0, 2.0, .01)
    
    xInit = np.array([[1., 0.]]).T
    xHist, timeHist = system.simulate(100, xInit)
        
    system.plot_hist(xHist, timeHist)
    
    