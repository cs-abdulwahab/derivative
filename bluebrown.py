# import pandas_profiling
import tensorflow as tf
import numpy as np


def f(x):
    return 2 * x


X = tf.random.normal(shape=(5,), seed=10, dtype=tf.float32, name="X")
X = tf.Variable(X)

print(f(X))
