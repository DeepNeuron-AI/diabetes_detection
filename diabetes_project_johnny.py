#Loading libraries for use
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split

#======Number of Neurons===========
inputLayer_size = 9
hiddenLayer1_size = 8 
hiddenLayer2_size = 4
outputLayer_size = 2 
#==================================

data = pd.read_csv('data.csv', )
data['result'].replace(['tested_positive', 'tested_negative'], [1, 0], inplace = True)

train_x, val_x, train_y, val_y = train_test_split(data.drop(['result'], axis = 1), data['result'], test_size = 0.2)

for insample in range(len(train_y)):
    print(np.expand_dims(train_x[insample], axis = 0))

#_input = tf.placeholder("float", shape=[None, inputLayer_size])
#_label = tf.placeholder("float", shape=[None, 1])
#
#first = tf.contrib.layers.fully_connected(_input, inputLayer_size, activation_fn=None)
#hidden1 = tf.contrib.layers.fully_connected(first, hiddenLayer1_size, activation_fn=None)
#hidden2 = tf.contrib.layers.fully_connected(hidden1, hiddenLayer2_size, activation_fn=None)
#output = tf.contrib.layers.fully_connected(hidden2, outputLayer_size, activation_fn=None)
#
#loss = tf.squeeze(tf.square(_label - output))
#backprop = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#
#print(train_x)

#for epoch in range(20):
#    for insample in range(len(train_y)):
#        X = np.expand_dims(train_x[insample], axis = 0) #expanding dimension
#        Y = [[train_y[insample]]]
#        sess.run(backprop, feed_dict={_input:X, _label:Y})

#print("hello")







# filenames = ["data.csv"]
# record_defaults = [tf.float32] * 9   # Eight required float columns
# dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header = True)
#
# print(dataset)
