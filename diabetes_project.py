#Loading libraries for use
import numpy as np
import tensorflow as tf


filenames = ["data.csv"]
record_defaults = [tf.float32] * 9   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)

print('ello')
