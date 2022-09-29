import numpy as np
import tensorflow as tf
import tensorflow.keras


def c2F(c):
    return c * 1.8 + 32


C = tf.random.uniform(shape=(50,), minval=-40, maxval=40, seed=1, name="C")
F = np.array(c2F(C))


l0 = tf.keras.layers.Dense(units=1, input_shape=(1,))

model = tf.keras.models.Sequential(l0)

model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(x = C , y=F , epochs=500)


