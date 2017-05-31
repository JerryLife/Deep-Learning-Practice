import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = []
for i in range(1, 200):
    x = np.random.normal(0.0, 0.6)
    dy = np.random.normal(0.0, 0.06)
    data.append((x, 0.5 * x + 0.9 + dy))
xs = [v[0] for v in data]
ys = [v[1] for v in data]

k = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = k * xs + b
loss = tf.reduce_mean(tf.square(y - ys))    # square mean of (y-ys)
optimizer = tf.train.GradientDescentOptimizer(0.5)
training = optimizer.minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(20):
        sess.run(training)
        print(step, sess.run(k), sess.run(b))
