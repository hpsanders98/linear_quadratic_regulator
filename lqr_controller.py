import numpy as np
import matplotlib.pyplot as plt
from control import lqr
from mass_spring_damper import MassSpringDamper
import cv2

class LQRController:
    
    def __init__(self, system, Q, R):
        
        self.system = system
        self.dt = self.system.dt
        self.K = self.calculate_K(system.A, system.B, Q, R)
        
    def calculate_K(self, A, B, Q, R):
        
        K, S, E = lqr(A, B, Q, R)
        
        return K
        
    def simulate(self, totalTime, x, xRef):
        
        numForwardSimulationSteps = int(totalTime / self.dt)
        elapsedTime = 0
        xHist = np.zeros((numForwardSimulationSteps, 2))
        timeHist = np.zeros(numForwardSimulationSteps)
        uHist = np.zeros(numForwardSimulationSteps)
        
        for i in range(0, numForwardSimulationSteps):
            
            u = -self.K @ (x - xRef)
            x = self.system.forward_simulate(x, u)
                        
            elapsedTime = elapsedTime + self.dt
            
            xHist[i, :] = x.flatten()
            timeHist[i] = elapsedTime
            uHist[i] = u
            
        return xHist, timeHist, uHist
    
    def plot_hist(self, x, totalTime, u):
        
        self.system.plot_hist(x, totalTime, u)
        
if __name__ == "__main__":
    
    system = MassSpringDamper(5.0, 1.0, 2.0, .01)
    Q = np.diag([20.0, 1.0])
    R = .001 * np.ones((1, 1))    
    controller = LQRController(system, Q, R)    
    xInit = np.array([[0., 0.]]).T
    xRef = np.array([[10.0, 0.0]]).T
    
    xHist, timeHist, uHist = controller.simulate(20, xInit, xRef)    
    controller.plot_hist(xHist, timeHist, uHist)
    
    