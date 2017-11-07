import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def make_data(t1, t2, d):
    dt = t2/d
    t_ = np.linspace(t1, t2, dt)
    data = np.array([t * np.sin(t) / 3 + 2 * np.sin(5 * t) for t in t_])
    t_next = t2 + d
    extra = t_next * np.sin(t_next) / 3 + 2 * np.sin(5 * t_next)
    return data, t_, extra

def next_batch(data, size, steps, n_inputs):
    batchX = []
    batchY = []
    x_ = np.array(np.zeros((steps, n_inputs)))
    y_ = np.array(np.zeros((steps, n_inputs)))
    for _ in range(size):
        for h in range(n_inputs):
            z = rnd.randint(0, data.__len__() - steps -1)
            x_[:,h] = data[z:z+steps]
            y_[:,h] = data[z+1:z+steps+1]
            batchX.append(x_)
            batchY.append(y_)
    return np.array(batchX), np.array(batchY)

def get_last(data, steps, n_inputs):
    batchX = []
    x_ = np.array(np.zeros((steps, n_inputs)))
    for h in range(n_inputs):
        s = data.__len__()-steps
        x_[:, h] = data[s:]
        batchX.append(x_)
    return np.array(batchX)

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                                                          activation=tf.nn.relu),
                                                                            output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_iterations = 1
batch_size = 5
dt = 0.5
data, t, target = make_data(0,30,dt)

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(data,batch_size,n_steps, n_inputs)
        sess.run(training_op, feed_dict={X: X_batch, Y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, Y: y_batch})
            print(iteration, "\tMSE:", mse)

    X_last = get_last(data, n_steps, n_inputs)
    y_pred = sess.run(outputs, feed_dict={X: X_last})
    last = y_pred[0].__len__()-1

    t_add = t[t.__len__()-1] + dt
    y_add = y_pred[0,last]
    plt.plot(t_add, y_add, 'ro')
    plt.plot(t_add, target, 'b^')
    plt.plot(t, data)
    plt.grid(True)
    plt.show()