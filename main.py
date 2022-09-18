import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

from tensorflow import Variable

x = tf.Variable(10.0)


def f(x):
    return x * 2


with tf.GradientTape() as tape:
    # tape.watch(x)
    y = f(x)

dy_dx = tape.gradient(y, x)

w = tf.Variable([[1.], [2.]])
x = tf.constant([[3., 4.]])
tf.matmul(w, x)
tf.sigmoid(w + x)

A1 = tf.constant([1, 2, 3, 4])
B1 = tf.constant([3, 4, 5, 5])
C1 = tf.multiply(A1, B1)

np.random.seed(101)
tf.random.set_seed(101)

X = np.linspace(1, 50)
Y = np.linspace(1, 50)

# Adding noise to the random linear data
X += np.random.uniform(-4, 4, 50)
Y += np.random.uniform(-4, 4, 50)

sns.scatterplot(data={"X": X, "Y": Y}, x="x_var", y="y_var")
plt.show()


W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

learning_rate = 0.01
training_epochs = 1000

# Hypothesis
y_pred = tf.add(tf.multiply(X, W), b)

n = len(x)  # Number of data points
# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
