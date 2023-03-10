import numpy as np
import matplotlib.pyplot as plt

class MassSpringDamper:
    
    def __init__(self, m, b, k, dt):
        
        self.dt = dt
        self.uMin = -10.0
        self.uMax = 10.0
        
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
    
    def visualize(self, x):
        
        x = x.reshape(-1, 1)

        l_box = 0.5
        h_box = 0.5
        h_wall = 1.0
        l_gap = 1.0
        l_plate = (l_gap + l_box)*1.5
        n_coils = 6
        h_coils = 0.2

        xf = x[0].astype(float)


        box_x = [xf, xf, xf+l_box, xf+l_box, xf]
        box_y = [0, h_box, h_box, 0, 0]

        base_x = [l_plate-l_gap, -l_gap, -l_gap, -l_gap]
        base_y = [0, 0, h_wall, h_box/2]

        spacing = (l_gap + xf)/float(n_coils*2)

        add = np.arange(-l_gap, xf, spacing)

        for i in range(n_coils*2):
            if i%2 == 0:
                base_y.append(h_box/2.0 - h_coils/2.0)
            else:
                base_y.append(h_box/2.0 + h_coils/2.0)
            base_x.append(add[i])

        base_x.append(xf)
        base_y.append(h_box/2.0 - h_coils/2.0)                    

        plt.clf()
        ax = plt.gca()
        plt.plot(base_x, base_y, 'k')
        plt.plot(box_x, box_y, 'b')
        plt.axis([-2, 2, -2+h_wall/2.0, 2+h_wall/2.0])
        plt.ion()
        plt.show()
        plt.pause(.0000001)
    
    def plot_hist(self, x, totalTime, u):
        
        fig, ax = plt.subplots(2)
        
        ax[0].plot(totalTime, x[:, 0], totalTime, x[:, 1])
        ax[0].set_ylabel('States')
        ax[0].legend(['x (m)', 'xdot (m/s)'])
        ax[0].grid()
        
        ax[1].plot(totalTime, u)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Input (N)')
        ax[1].grid()
        
        plt.show()
    
    