import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

class RobotArm:
    def __init__(self, links=[50, 40, 40], target_pos=[0, 0]):
        # Robot link length parameter
        self.links = links
        self.target_pos = target_pos

    def rotateZ(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return rz

    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return t

    # Forward Kinematics
    def FK(self, joints_angle):
        n_links = len(self.links)
        P = []
        P.append(np.eye(4))
        for i in range(0, n_links):
            R = self.rotateZ(joints_angle[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P

    # Here is objective function
    # GA will minimize this function
    def calc_distance_error(self, joints_angle):
        P = self.FK(joints_angle)
        current_pos = [float(P[-1][0, 3]), float(P[-1][1, 3])]
        error = euclidean(current_pos, self.target_pos)
        return error

    # Plot joint configuration result
    def plot(self, joints_angle):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        P = self.FK(joints_angle)
        for i in range(len(self.links)):
            start_point = P[i]
            end_point = P[i + 1]
            ax.plot([start_point[0, 3], end_point[0, 3]], [start_point[1, 3], end_point[1, 3]], linewidth=5)
        plt.grid()
        plt.show()
