# import pandas_profiling
import tensorflow as tf
import numpy as np
import math

#
# def f(x):
#     return math.sin(x)
#
#
# def g(x):
#     return  f(x) ** 3
#
#
# x = tf.Variable(3.0, dtype="float")
# with tf.GradientTape() as tape:
#     y = f(x)

w = tf.Variable(tf.random.normal(shape=(3, 2), dtype=tf.float32, seed=1), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)

print(loss)
