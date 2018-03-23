#!/usr/bin/python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Data generator
def get_sample():
    # y = 0.2x + 0.3
    x_gen = np.random.random()
    y_gen = 0.2 * x_gen + 0.3
    return np.reshape(x_gen, [1, 1]), np.reshape(y_gen, [1, 1])

# Parameters
learning_rate = 0.01
num_samples = 10000

# Input and output of the network
x = tf.placeholder(tf.float32, [1, 1])
y = tf.placeholder(tf.float32, [1, 1])


# Network definition
weight = tf.Variable(tf.random_normal([1, 1]))
bias = tf.Variable(tf.random_normal([1]))
y_pred = tf.add(tf.matmul(x, weight), bias)

# Optimizer and cost function
cost = tf.squared_difference(y_pred, y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    sample = 0
    costs = list()
    while sample < num_samples:
        sample += 1
        train_x, train_y = get_sample()
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
        costs.append(c.tolist()[0])
        if sample % 1000 == 0:
            print("Cost -", c)
    print("\nFinal weight and bias (m and c)")
    print("W -", weight.eval(), ", B -", bias.eval())

plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Samples')
plt.savefig("cost-sample.png")
