import numpy as np
import tensorflow as tf
import seaborn as sns


def func(x, y):
    return x + y  # draw  a 3D Graph of this function

#
# arr1 = np.linspace(1, 5, 5)
# arr2 = np.linspace(1, 5, 5)
# x = tf.Variable(arr1, dtype="float")
# y = tf.Variable(arr2, dtype="float")
#
# tf.meshgrid(x,y)
#
# with tf.GradientTape() as g:
#     z = func(x, y)
#
# dy_dx = g.gradient(z, [x, y])
#
# # xs += np.random.uniform(1,10,len(xs))
