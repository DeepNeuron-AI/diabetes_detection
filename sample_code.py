import numpy as np
import tensorflow as tf

inputLayer_size = 17
hiddenLayer2_size = 20


data = np.random.rand(500, inputLayer_size)
labels = []
for row in data:
    if np.sum(row[0:5])-np.sum(row[5:inputLayer_size - 1]) > 0:
        labels.append(1)
    else:
        labels.append(-1)

#print(data)
#print(labels[0:5])

_input = tf.placeholder("float", shape=[None, inputLayer_size])
_label = tf.placeholder("float", shape=[None, 1])

hidden1 = tf.contrib.layers.fully_connected(_input, inputLayer_size, activation_fn=None)
hidden2 = tf.contrib.layers.fully_connected(hidden1, hiddenLayer2_size, activation_fn=None)
output = tf.contrib.layers.fully_connected(hidden2, 1, activation_fn=None)

loss = tf.squeeze(tf.square(_label - output))
backprop = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(20):
    for insample in range(400):
        X = np.expand_dims(data[insample], axis = 0) #expanding dimension
        Y = [[labels[insample]]]
        sess.run(backprop, feed_dict={_input:X, _label:Y})



accuracy = 0
meanLoss = 0
for insample in range(400, 500):
    X = np.expand_dims(data[insample], axis = 0)
    Y = [[labels[insample]]]
    result, L = sess.run((output, loss), feed_dict={_input: X, _label: Y})
    meanLoss += L
    if result > 0 and labels[insample] == 1:
        accuracy += 1
    elif result < 0 and labels[insample] == -1:
        accuracy += 1

print(accuracy)
print(meanLoss)
