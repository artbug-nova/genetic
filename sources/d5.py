import numpy as np

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

        reduce_to_nil = lambda n: np.cond(n > 0,
                                          lambda: np.constant(0, dtype=np.float32), lambda: np.abs(n))

        return W1 * np.reduce_sum(
            np.map_fn(reduce_to_nil, np.subtract(Q, self.penaltiesMin))
        ) + W2 * np.reduce_sum(np.map_fn(reduce_to_nil, np.subtract(self.penaltiesMax, Q)))

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
Z4 = KinematicPart(0, 0, np.pi / 2, bmin=180 * r, bmax=180 * r)
Z5 = KinematicPart(0, 104.9, np.pi / 2, bmin=-5 * r, bmax=15 * r)

parts = [Z1, Z2, Z3, Z4, Z5]#, Z6]

RV = Robot(parts)


Q01 = 0 * r
Q12 = 1.5858247741421216
Q23 = -3.113309451245733
Q34 = 0 * r
Q45 = 0 * r


Q0 = [Q01, Q12, Q23, Q34, Q45]

print(RV.getXYZ(Q0))


# [[ 7.71269882e+00]
#  [-8.98878949e-16]
#  [ 2.85320193e+02]]