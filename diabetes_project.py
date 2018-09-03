#Loading libraries for use
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split

data = pd.read_csv('data.csv')
print(data.head(10))
data['result'].replace(['tested_positive', 'tested_negative'], [1, 0], inplace = True)
print(data.head(10))
# filenames = ["data.csv"]
# record_defaults = [tf.float32] * 9   # Eight required float columns
# dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header = True)
#
# print(dataset)
