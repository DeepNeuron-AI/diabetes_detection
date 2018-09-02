import numpy as np
import tensorflow as tf

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0], [0], [0], [0], [0], [0], [0], [0], [""]]
    preg, plas, pres, skin, insu, mass, pedi, age, result = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([preg, plas, pres, skin, insu, mass, pedi, age])
    return features, result

filename_queue = tf.train.string_input_producer('data.csv', num_epochs=1, shuffle=False)
features, result = create_file_reader_ops(filename_queue)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while True:
        try:
            example_data, country_name = sess.run([example, country])
            print(example_data, country_name)
        except tf.errors.OutOfRangeError:
            break
