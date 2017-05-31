import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

data_board = []
size = 2000
for i in range(size):
    rand = np.random.random()
    if rand > 0.6:
        x = np.random.normal(3.0, 0.8)
        y = np.random.normal(1.0, 0.5)
        data_board.append((x, y))
    elif rand < 0.3:
        x = np.random.normal(1.0, 0.4)
        y = np.random.normal(2.0, 0.9)
        data_board.append((x, y))
    else:
        x = np.random.normal(3.0, 0.5)
        y = np.random.normal(3.0, 0.5)
        data_board.append((x, y))

xs = [v[0] for v in data_board]
ys = [v[1] for v in data_board]
plt.plot(xs, ys, 'bo', label='Original data')
plt.show()
plt.close()

vectors = tf.constant(data_board)
k = 3
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

# linear mathematics
expand_vectors = tf.expand_dims(vectors, 0)         # ? * size=2000 * k=4
expand_centroids = tf.expand_dims(centroids, 1)     # k=4 * ? * 2
diff = tf.subtract(expand_vectors, expand_centroids)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignment = tf.arg_min(distances, 0)

mean_data = []
for c in range(k):
    selected = tf.reshape(tf.where(tf.equal(assignment, c)), [1, -1])
    new_mean = tf.reduce_mean(tf.gather(vectors, selected), reduction_indices=[1])
    mean_data.append(new_mean)

means = tf.concat(mean_data, 0)
new_centroids = tf.assign(centroids, means)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        _, centroids_values, assignment_values = sess.run(
            [new_centroids, centroids, assignment])     # run 3 tensors
    print(centroids_values)
    colors = []
    x = []
    y = []
    for i in range(len(assignment_values)):
        x.append(data_board[i][0])
        y.append(data_board[i][1])
        colors.append(assignment_values[i])
    plt.scatter(x, y, c=colors)
    plt.plot(label='Original data')
    plt.show()
