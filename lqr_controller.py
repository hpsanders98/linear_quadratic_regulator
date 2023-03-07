import numpy as np
import matplotlib.pyplot as plt
from control import lqr
from mass_spring_damper import MassSpringDamper

class LQRController:
    
    def __init__(self, system, Q, R):
        
        self.system = system
        self.dt = self.system.dt
        self.K = self.calculate_K(system.A, system.B, Q, R)
        
    def calculate_K(self, A, B, Q, R):
        
        K, S, E = lqr(A, B, Q, R)
        
        return K
        
    def simulate(self, totalTime, x):
        
        numForwardSimulationSteps = int(totalTime / self.dt)
        elapsedTime = 0
        xHist = np.zeros((numForwardSimulationSteps, 2))
        timeHist = np.zeros(numForwardSimulationSteps)
        
        for i in range(0, numForwardSimulationSteps):
            
            u = -self.K @ (x - np.array([[1.5, 0.0]]).T)
            x = self.system.forward_simulate(x, u)
                        
            elapsedTime = elapsedTime + self.dt
            
            xHist[i, :] = x.flatten()
            timeHist[i] = elapsedTime
            
        return xHist, timeHist
    
    def plot_hist(self, x, totalTime):
        
        plt.plot(totalTime, x[:, 0])
        plt.show()
        
if __name__ == "__main__":
    
    system = MassSpringDamper(50.0, 10.0, 2.0, .01)
    Q = np.ones((2, 2))
    R = .01 * np.ones((1, 1))    
    controller = LQRController(system, Q, R)
    
    xInit = np.array([[10., 0.]]).T
    xHist, timeHist = controller.simulate(100, xInit)
    
    controller.plot_hist(xHist, timeHist)
    
    