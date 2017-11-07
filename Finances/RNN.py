import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd

def next_batch(data, size, steps, n_inputs, shift, iter):
    batchX = []
    batchY = []
    x_ = np.array(np.zeros((steps, n_inputs)))
    y_ = np.array(np.zeros((steps, n_inputs)))
    for i in range(size):
        rnd.seed = i+iter
        for h in range(n_inputs):
            z = rnd.randint(0, len(data)-steps-shift)
            x_[:,h] = data[z:z+steps]
            y_[:,h] = data[z+shift:z+steps+shift]
            batchX.append(x_)
            batchY.append(y_)
    return np.array(batchX), np.array(batchY)

def get_last(data, steps, n_inputs, shift):
    batchX = []
    x_ = np.array(np.zeros((steps, n_inputs)))
    for h in range(n_inputs):
        s = len(data)-steps-shift
        d = data[s:len(data)-shift]
        x_[:, h] = d
        batchX.append(x_)
    return np.array(batchX)

n_steps = 120
n_inputs = 1
n_neurons = 270
n_outputs = 1

shift = 6

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

n_iterations = 1000
batch_size = 128

data = pd.read_csv('All_data.csv',parse_dates=[0])

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(data['V'],batch_size,n_steps,n_inputs,shift, iteration)
        sess.run(training_op, feed_dict={X: X_batch, Y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, Y: y_batch})
            print(iteration, "\tMSE:", mse)

    X_last = get_last(data['V'], n_steps, n_inputs,shift)
    y_pred = sess.run(outputs, feed_dict={X: X_last})

    df = pd.DataFrame(columns=['Date','V','Forecast'])
    df['Date'] = data['Date'][-shift:]
    df['V'] = data['V'][-shift:]
    df['Forecast'] = y_pred[0][-shift:]

    df.set_index("Date",inplace=True)
    df['V'].plot(legend='V')
    df['Forecast'].plot(legend='Forecast',marker='o',color='r')

    plt.grid(True)
    plt.show()
