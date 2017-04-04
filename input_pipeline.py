import tensorflow as tf

import os


# Shape of inputs, tuple of width x height x depth
INPUTS_SHAPE = (9, 9, 1)
# Number of examples to generate if data files don't exist
NUM_EXAMPLES = 10000
NUM_CLASSES = 2
# Used for calculating the queue capacity when feeding data into the training runner
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


def read_and_decode(filename_queue, scope=None):
    '''Read and decode 'tfrecord' binary files as defined in utils module

    Args:
        filename_queue: a queue of filenames generated from 
            tf.train.string_input_producer
        scope (optional): define a new name scope 

    Returns:
        result: Record class with width, height, depth, label and inputs
            properties.
    '''
    with tf.name_scope(scope or 'read_and_decode'):
        class Record(object):
            '''Placeholder class for a record'''
            pass

        result = Record()
        label_bytes = 1
        result.width = INPUTS_SHAPE[0]
        result.height = INPUTS_SHAPE[1]
        result.depth = INPUTS_SHAPE[2]
        inputs_bytes = result.width * result.height * result.depth

        # Length of record bytes in the dataset
        # Defined in utils module
        record_bytes = label_bytes + inputs_bytes

        reader = tf.TFRecordReader()
        result.key, serialized_example = reader.read(filename_queue)

        feature_map = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string)
        }
        parsed = tf.parse_single_example(serialized_example, feature_map)
        record_bytes = tf.decode_raw(parsed['image/encoded'], tf.int8)

        # first byte is the label
        result.label = tf.cast(
            tf.strided_slice(record_bytes, begin=[0], end=[label_bytes]), tf.int32)

        # remaining bytes is the inputs
        result.uint8inputs = tf.reshape(
            tf.strided_slice(record_bytes, begin=[label_bytes], 
                end=[label_bytes + inputs_bytes]), 
            [result.width, result.height, result.depth])

    return result


def _generate_batch(features, labels, min_queue_examples, batch_size, scope=None, shuffle=True):
    '''

    '''
    num_preprocess_threads = 4
    if shuffle:
        example_batch, label_batch = tf.train.shuffle_batch(
            [features, labels],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False,
            min_after_dequeue=min_queue_examples)
    else:
        example_batch, label_batch = tf.train.batch(
            [features, labels],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=False)

    return example_batch, tf.reshape(label_batch, [batch_size])


def train_inputs(data_dir, batch_size):
    # Name scope inputs to beautify Tensorboard graph
    with tf.name_scope('train_inputs') as scope:
        filenames = [os.path.join(data_dir, 'data_batch_%d.tfrecords' % i)
                     for i in range(1, 2)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        filename_queue = tf.train.string_input_producer(filenames)   
        read_input = read_and_decode(filename_queue, scope)

        # set shapes of tensors
        reshaped_inputs = tf.cast(read_input.uint8inputs, tf.float32)
        reshaped_inputs.set_shape([INPUTS_SHAPE[0], INPUTS_SHAPE[1], INPUTS_SHAPE[2]])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    # return _generate_batch(features, labels, batch_size, scope, shuffle=True)
    return _generate_batch(reshaped_inputs, read_input.label, 
                           min_queue_examples, batch_size, 
                           scope, shuffle=True)


def inputs(data_dir, eval_data, batch_size):
    with tf.name_scope('inputs') as scope:
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.tfrecords' % i)
                         for i in range(1, 2)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        filename_queue = tf.train.string_input_producer(filenames)
        read_input = read_and_decode(filename_queue, scope)

        reshaped_inputs = tf.cast(read_input.uint8inputs, tf.float32)
        reshaped_inputs.set_shape([INPUTS_SHAPE[0], INPUTS_SHAPE[1], INPUTS_SHAPE[2]])
        read_input.label.set_shape([1])
        
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    return _generate_batch(reshaped_inputs, read_input.label, 
                           min_queue_examples, batch_size, 
                           scope, shuffle=False)
