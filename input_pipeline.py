import tensorflow as tf

import os


INPUTS_SIZE = (9, 1, 1)
NUM_CLASSES = 2


def read_and_decode(filename_queue, scope=None):
        # read
        with tf.name_scope(scope or 'read_and_decode'):
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
            width = INPUTS_SIZE[0]
            height = INPUTS_SIZE[1]
            depth = INPUTS_SIZE[2]

            features = tf.decode_raw(parsed['x'], tf.int64)
            features = tf.reshape(features, [width, height, depth])
            features = tf.cast(features, dtype=tf.float32)
            labels = parsed['y']
            labels = tf.one_hot(labels, 2, on_value=1, dtype=tf.int32)

        return features, labels


def _generate_batch(features, labels, batch_size, scope=None, shuffle=True):
    min_after_dequeue = 10000
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
    # Name scope inputs to beautify Tensorboard graph
    with tf.name_scope('train_inputs') as scope:
        filenames = [os.path.join(data_dir, 'data_batch_%d.tfrecords' % i)
                     for i in range(1, 2)]
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        filename_queue = tf.train.string_input_producer(filenames)
        features, labels = read_and_decode(filename_queue, scope)

    return _generate_batch(features, labels, batch_size, scope, shuffle=True)


def inputs(data_dir, eval_data, batch_size):
    with tf.name_scope('inputs') as scope:
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
