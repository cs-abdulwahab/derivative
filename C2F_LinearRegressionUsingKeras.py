import numpy as np
import tensorflow as tf
import tensorflow.keras


def c2F(c):
    return c * 1.8 + 32


C = tf.random.uniform(shape=(50,), minval=-40, maxval=40, seed=1, name="C")
F = np.array(c2F(C))
