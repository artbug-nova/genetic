import tensorflow as tf


# Класс, представляющий собой одну кинематическую пару
class KinematicPart:
    s = tf.constant(0, dtype=tf.float32)
    a = tf.constant(0, dtype=tf.float32)
    alpha = tf.constant(0, dtype=tf.float32)

    borderMin = tf.constant(0, dtype=tf.float32)
    borderMax = tf.constant(0, dtype=tf.float32)

    def __init__(self, s, a, alpha, bmin, bmax):
        self.s = tf.constant(s, dtype=tf.float32)
        self.a = tf.constant(a, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.borderMin = tf.constant(bmin, dtype=tf.float32)
        self.borderMax = tf.constant(bmax, dtype=tf.float32)

    def getMatrix(self, q):
        return [
            [tf.cos(q), -tf.sin(q) * tf.cos(self.alpha), tf.sin(q) * tf.sin(self.alpha), self.a * tf.cos(q)],
            [tf.sin(q), tf.cos(q) * tf.cos(self.alpha), -tf.cos(q) * tf.sin(self.alpha), self.a * tf.sin(q)],
            [0, tf.sin(self.alpha), tf.cos(self.alpha), self.s],
            [0, 0, 0, 1]]
