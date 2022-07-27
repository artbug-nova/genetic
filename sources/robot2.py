import tensorflow as tf

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

        reduce_to_nil = lambda n: tf.cond(n > 0,
                                          lambda: tf.constant(0, dtype=tf.float32), lambda: tf.abs(n))

        return W1 * tf.reduce_sum(
            tf.map_fn(reduce_to_nil, tf.subtract(Q, self.penaltiesMin))
        ) + W2 * tf.reduce_sum(tf.map_fn(reduce_to_nil, tf.subtract(self.penaltiesMax, Q)))

    '''
    Получить координаты схвата (конечного звена)
    '''

    def getXYZ(self, Q):
        return self.getXYZPair(Q, len(self.parts))[:3]

    '''
    Получить координаты конкретной пары 
    '''

    def getXYZPair(self, Q, pair):

        resultMatrix = tf.eye(4, dtype=tf.float32)

        for i, p in enumerate(self.parts):

            if i == pair:
                break

            resultMatrix = tf.matmul(resultMatrix, p.getMatrix(Q[i]))

        xyz1 = tf.matmul(resultMatrix, tf.constant([[0], [0], [0], [1]], dtype=tf.float32))

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