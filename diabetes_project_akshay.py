"""
This module learns the diabetes set to predict whether a person has diabetes
Network: MLP
LearningRate(Initial): 0.005
Backprop: Descent (ReLu for later)
"""
import tensorflow as tf
import numpy as np
#np.warnings.filterwarnings('ignore')    # For NaN entries
input_layer_size = 8
data = np.genfromtxt("data_scratch.csv", delimiter=",")
data_feed = np.zeros(shape=(768, input_layer_size))
for i in range(768):
    for j in range(input_layer_size):
        data_feed[i][j] = data[i][j]
labels = data[:, 9]

# Creating the MLP TF graph
_input = tf.placeholder("float", shape=[None, input_layer_size])
_label = tf.placeholder("float", shape=[None, 1])

hiddenLayer1_size = 8
hidden1 = tf.contrib.layers.fully_connected(_input, hiddenLayer1_size, activation_fn=None)
hiddenLayer2_size = 4
hidden2 = tf.contrib.layers.fully_connected(hidden1, hiddenLayer2_size, activation_fn=None)

output = tf.contrib.layers.fully_connected(hidden2, 1, activation_fn=None)
loss = tf.squeeze(tf.square(_label - output))
backprop = tf.train.GradientDescentOptimizer(0.05).minimize(loss)  # 0.005 is the L learning rate

# Training the MLP
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(10):
    for insample in range(600):  # 600 for training 168 for testing
        X = np.expand_dims(data_feed[insample], axis=0)
        Y = [[labels[insample]]]
        sess.run(backprop, feed_dict={_input: X, _label: Y})

# Testing the MLP
accuracy = 0
meanLoss = 0

for insample in range(600, 768):
    X = np.expand_dims(data_feed[insample], axis=0)
    Y = [[labels[insample]]]
    result, L = sess.run((output, loss), feed_dict={_input: X, _label: Y})
    meanLoss += L
    if result >= 0.5 and labels[insample] == 1.0:
        accuracy += 1
    elif result < 0.5 and labels[insample] == 0.0:
        accuracy += 1

accuracy /= 100
meanLoss /= 100
print("Accuracy = ", accuracy, ", ", "mean Loss = ", meanLoss)
