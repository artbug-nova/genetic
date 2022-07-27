import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class Robot:
    parts = []
    penaltiesMin = None
    penaltiesMax = None

    def __init__(self, parts):
        self.parts = parts
        self.penaltiesMin = [(p.borderMin) for p in self.parts]
        self.penaltiesMax = [(p.borderMax) for p in self.parts]

    '''
    Получить значение штрафа для данных обобщенных координат
    '''


    def penalty(self, Q, W1=1, W2=1):

        #lambda (t): (200 * exp(-t)) if t > 200 else (400 * exp(-t))
        reduce_to_nil = lambda n: 0 if n > 0 else np.abs(n)

        subtract = np.subtract(Q, self.penaltiesMin)
        np_subtract = np.subtract(self.penaltiesMax, Q)
        return W1 * np.sum(list(map(reduce_to_nil, subtract))) \
               + W2 * np.sum(list(map(reduce_to_nil, np_subtract)))

    '''
    Получить координаты схвата (конечного звена)
    '''

    def getXYZ(self, Q):
        return self.getXYZPair(Q, len(self.parts))[:3]

    '''
    Получить координаты конкретной пары 
    '''

    def getXYZPair(self, Q, pair):

        resultMatrix = np.eye(4, dtype=np.float32)

        for i, p in enumerate(self.parts):

            if i == pair:
                break

            resultMatrix = np.matmul(resultMatrix, p.getMatrix(Q[i]))

        xyz1 = np.matmul(resultMatrix, [[0], [0], [0], [1]])

        return xyz1

    '''
    Массив координат всех пар (для построения графика)
    '''

    def getPairPoints(self, Q):

        result = []

        for i, p in enumerate(self.parts):
            pairXYZ = self.getXYZPair(Q, i)
            result.append([pairXYZ[0], pairXYZ[1], pairXYZ[2]])

        return result




class KinematicPart:
    s = 0
    a = 0
    alpha = 0

    borderMin = 0
    borderMax = 0

    def __init__(self, s, a, alpha, bmin, bmax):
        self.s = s
        self.a = a
        self.alpha = alpha
        self.borderMin = bmin
        self.borderMax = bmax

    def getMatrix(self, q):
        return [
            [np.cos(q), -np.sin(q) * np.cos(self.alpha), np.sin(q) * np.sin(self.alpha), self.a * np.cos(q)],
            [np.sin(q), np.cos(q) * np.cos(self.alpha), -np.cos(q) * np.sin(self.alpha), self.a * np.sin(q)],
            [0, np.sin(self.alpha), np.cos(self.alpha), self.s],
            [0, 0, 0, 1]]


r = np.pi / 180.0

#
Z1 = KinematicPart(300, 0, np.pi / 2, bmin=-185 * r, bmax=185 * r)
Z2 = KinematicPart(0, 250, 0, bmin=50 * r, bmax=270 * r)
Z3 = KinematicPart(0, 160, 0, bmin=-360 * r, bmax=360 * r)
Z4 = KinematicPart(0, 0, np.pi / 2, bmin=-10 * r, bmax=10 * r)
Z5 = KinematicPart(0, 104.9, np.pi / 2, bmin=0 * r, bmax=0 * r)

parts = [Z1, Z2, Z3, Z4, Z5]#, Z6]

RV = Robot(parts)


Q01 = 0 * r
Q12 = 90 * r
Q23 = 270 * r
Q34 = 0 * r
Q45 = 0 * r


Q0 = [Q01, Q12, Q23, Q34, Q45]

def loss_function(Q0, target):
    xyz = RV.getXYZ(Q0)
    penalty = RV.penalty(Q0, 1, 1)
    powers = (target - xyz) ** 2
    return np.sum(np.sqrt(powers)) + penalty
    #return euclidean(target, xyz) + penalty

traj = [
    [[264], [0], [550.9]],
    [[263], [0], [552]],
    [[265], [0], [550]],
    [[266], [0], [549]],
    [[263], [0], [552]],
    # [[264.6], [0], [550.4]],
    # [[264.4], [0], [550.6]],
    # [[264.2], [0], [550.8]],
    # [[265], [0], [552]],
]
N = 50

j = 0
for val in traj:
    target = val
    looses = 11
    delta_q = 0.02 * r
    delta = [0, delta_q, 0, 0, 0]
    ik = 1
    while looses > 0.2:
        function = loss_function(Q0, target)
        if ik % 3 == 0:
            delta = [0, 0, 0, delta_q, 0]
        elif ik % 2 == 0:
            delta = [0, 0, delta_q, 0, 0]
        else:
            delta = [0, delta_q, 0, 0, 0]
        n = loss_function(np.add(Q0, delta), target)

        prir = (n - function) / delta_q
        if n < 0.2:
            looses = n
            if ik % 3 == 0:
                Q0[3] = Q0[3] + delta_q
            elif ik % 2 == 0:
                Q0[2] = Q0[2] + delta_q
            else:
                Q0[1] = Q0[1] + delta_q

            xyz = RV.getXYZ(Q0)
            print(xyz)

            xyz1 = RV.getXYZPair(Q0, 1)
            xyz2 = RV.getXYZPair(Q0, 2)
            xyz3 = RV.getXYZPair(Q0, 3)
            xyz4 = RV.getXYZPair(Q0, 4)

            fig, ax = plt.subplots()
            ax.plot([xyz1[0], xyz2[0], xyz3[0], xyz4[0], xyz[0]], [xyz1[2], xyz2[2], xyz3[2], xyz4[2], xyz[2]])
            ax.set_title('matplotlib.axes.Axes.plot() example 1')
            fig.canvas.draw()
            plt.grid()
            plt.show()

        else:
            f1 = 0.000025 * prir
            f2 = 0.000025 * prir
            f3 = 0.000025 * prir
            if ik % 3 == 0:
                Q0[3] = Q0[3] - f3
            elif ik % 2 == 0:
                Q0[2] = Q0[2] - f2
            else:
                Q0[1] = Q0[1] - f1

            looses = loss_function(Q0, target)
            if looses < function:
                print('#####################')
                print(looses)
                print('#####################')
                print(Q0)
                print('#####################')
                xyz = RV.getXYZ(Q0)
                print(xyz)

                # xyz1 = RV.getXYZPair(Q0, 1)
                # xyz2 = RV.getXYZPair(Q0, 2)
                # xyz3 = RV.getXYZPair(Q0, 3)
                # xyz4 = RV.getXYZPair(Q0, 4)
                #
                # fig, ax = plt.subplots()
                # ax.plot([xyz1[0], xyz2[0], xyz3[0], xyz4[0], xyz[0]], [xyz1[2], xyz2[2], xyz3[2], xyz4[2], xyz[2]])
                # ax.set_title('matplotlib.axes.Axes.plot() example 1')
                # fig.canvas.draw()
                # plt.grid()
                # plt.show()

            ik = ik + 1




# # Plot joint configuration result
# def plot(self, joints_angle):
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     P = self.FK(joints_angle)
#     for i in range(len(self.links)):
#         start_point = P[i]
#         end_point = P[i + 1]
#         ax.plot([start_point[0, 3], end_point[0, 3]], [start_point[1, 3], end_point[1, 3]], linewidth=5)
#     plt.grid()
#     plt.show()


