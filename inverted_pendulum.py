import numpy as np
import matplotlib.pyplot as plt

class InvertedPendulum:
    
    def __init__(self, m, b, l, dt):
        
        self.m = m
        self.b = b
        self.l = l
        self.I = self.m * self.l**2.0
        self.dt = dt
        self.g = 9.81
        self.uMin = -1.0
        self.uMax = 1.0
        
        self.A = np.matrix([[-self.b/self.I, 0],
                       [1.0, 0]])
        self.B = np.matrix([[1.0/self.I],
                       [0.0]])
        
    def calculate_xdot(self, x, u):
        
        thetaDot = x[0, 0]
        theta = x[1, 0]
        
        xdot = np.zeros(x.shape)
        xdot[0, :] = (-self.b*thetaDot + self.m * self.g*np.sin(theta) + u)/self.I
        xdot[1, :] = thetaDot
        return xdot
    
    def forward_simulate(self, x, u):
        
        xdot = self.calculate_xdot(x, u)
        
        x[1,:] = (x[1,:] + np.pi) % (2*np.pi) - np.pi        
        
        xNext = x + self.dt * xdot
        
        return xNext
    
    def visualize(self, x):

        x = x.reshape(-1, 1)
        CoM = [-0.5*np.sin(x[1, 0]), 0.5*np.cos(x[1, 0])]
        theta = x[1, 0]

        x = [CoM[0] + self.l/2.0 *
             np.sin(theta), CoM[0] - self.l/2.0*np.sin(theta)]
        y = [CoM[1] - self.l/2.0 *
             np.cos(theta), CoM[1] + self.l/2.0*np.cos(theta)]

        massX = CoM[0] - self.l/2.0*np.sin(theta)
        massY = CoM[1] + self.l/2.0*np.cos(theta)

        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.plot(x, y)
        plt.scatter(massX, massY, 50, 'r')
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.ion()
        plt.show()
        plt.pause(.0000001)
    
    def plot_hist(self, x, totalTime, u):
        
        fig, ax = plt.subplots(2)
        
        ax[0].plot(totalTime, x[:, 1], totalTime, x[:, 0])
        ax[0].set_ylabel('States')
        ax[0].legend(['Theta (rad)', 'ThetaDot (rad/s)'])
        ax[0].grid()
        
        ax[1].plot(totalTime, u)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Input (N)')
        ax[1].grid()
        
        plt.show()