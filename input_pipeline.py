import tensorflow as tf

import os


def read_and_decode(filename_queue):
        # read
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        feature_map = {
            'x': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
            'y': tf.FixedLenFeature(
                shape=[], dtype=tf.int64,
                default_value=None)
        }
        parsed = tf.parse_single_example(serialized_example, feature_map)

        # decode
        width = 5
        height = 1
        depth = 1

        features = tf.decode_raw(parsed['x'], tf.int64)
        features = tf.reshape(features, [width, height, depth])
        features = tf.cast(features, dtype=tf.float32)
        labels = parsed['y']

        return features, labels


def _generate_batch(features, labels, batch_size, shuffle):
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 + batch_size

    if shuffle:
        example_batch, label_batch = tf.train.shuffle_batch(
            [features, labels],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=4,
            allow_smaller_final_batch=False,
            min_after_dequeue=min_after_dequeue)
    else:
        example_batch, label_batch = tf.train.batch(
            [features, labels],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=4,
            allow_smaller_final_batch=False)

    return example_batch, label_batch


def train_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.tfrecords' % i)
                 for i in range(1, 2)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    # filename_queue = tf.train.string_input_producer(
    #     ['data/const_train_1_200000x5x1x1.tfrecords',
    #      'data/const_train_2_200000x5x1x1.tfrecords'], num_epochs=None)

    features, labels = read_and_decode(filename_queue)

    return _generate_batch(features, labels, batch_size, shuffle=True)


def inputs(data_dir, eval_data, batch_size):

    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.tfrecords' % i)
                     for i in range(1, 2)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.tfrecords')]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    features, labels = read_and_decode(filename_queue)
    return _generate_batch(features, labels, batch_size, shuffle=False)
