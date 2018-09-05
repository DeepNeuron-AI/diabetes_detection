"""
This module learns the diabetes set to predict whether a person has diabetes
Network: MLP
LearningRate(Initial): 0.005
Backprop: Descent (ReLu for later)
"""
import tensorflow as tf
import numpy as np

input_layer_size = 8
data = np.genfromtxt("data_scratch.csv", delimiter=",")
preg = data[:, 0]
plas = data[:, 1]
pres = data[:, 2]
skin = data[:, 3]
insu = data[:, 4]
mass = data[:, 5]
pedi = data[:, 6]
age = data[:, 7]
labels = data[:, 9]

# Creating the MLP TF graph
_input = tf.placeholder("float", shape=[None, input_layer_size])
_label = tf.placeholder("float", shape=[None, 1])

hiddenLayer1_size = 8
hidden1 = tf.contrib.layers.fully_connected(_input, hiddenLayer1_size, activation_fn = None)
hiddenLayer2_size = 4
hidden2 = tf.contrib.layers.fully_connected(hidden1, hiddenLayer2_size, activation_fn = None)

output = tf.contrib.layers.fully_connected(hidden2, 1, activation_fn = None)
loss = tf.squeeze(tf.square(_label - output))
backprop = tf.train.GradientDescentOptimizer(0.005).minimize(loss) # 0.005 is the L learning rate

# Training the MLP
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# for epoch in range(5):
#     for insample in range(400): # 400 for training 100 for testing
#             X = np.expand_dims(data[insample], axis=0)
#             Y = [[labels[insample]]]
#             sess.run(backprop, feed_dict={_input: X, _label: Y})


