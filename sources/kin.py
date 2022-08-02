import numpy as np


# Класс, представляющий собой одну кинематическую пару
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
