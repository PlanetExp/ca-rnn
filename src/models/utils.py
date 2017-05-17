'''Various utilities

Created 6 mars 2017
@author Frederick Heidrich
'''
import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle

import csv
import sys
import os

NUM_EXAMPLES = 10000


def get_connection_length(board, start_set=None, target_set=None):
    '''
        Determine the connectivity between the cell sets 'start_set' and
        'target_set'. That is, check for the existence of a path of connected stones
        from start_set to target_set.

        If start_set is None, it is initialized with the set of stones of row 0 (top row).
        if target_set is None, it is initialized with the set of stones of the bottom row.


        Return  connection_length
             connection_length is None  is the cell sets are not connected
             otherwise, it is the length of the connecting path

        Params
            board : uint8 np array,
                    zero -> background
                    non-zero -> stone
            start_set, target_set : cell coords  of the cell sets.sys.maxsize
                                    Lists of (row,col) pairs

    '''
    num_rows, num_cols = board.shape[0], board.shape[1]

    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .

    def getNeighbours(A, T):
        '''
            Return the set 'N' of stony neighbors of 'A', that are not in 'T' (taboo set)
            The neighbors are represented with the pairs of their (row, column) coordinates.
            Note that this is a sub-function.
        '''
        N = set()
        for r, c in A:
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                rn, cn = r + dr, c + dc
                if ((0 <= rn < num_rows) and (0 <= cn < num_cols) and board[rn, cn] != 0 and (rn, cn) not in T):
                    N.add((rn, cn))
        return N

    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .

    if start_set is None:
        start_set = [(0, c) for c in range(num_cols) if board[0, c] != 0]

    if target_set is None:
        target_set = [(num_rows - 1, c)
                      for c in range(num_cols) if board[num_cols - 1, c] != 0]

    g = 0  # generation index
    T = set()  # taboo cells
    F = set(start_set)  # frontier

    while not F.intersection(target_set):  # reached target?
        g += 1
        T.update(F)
        F = getNeighbours(F, T)
        if not F:  # cul de sac!
            return None
    return g


def board_generator(
        shape,
        min_connection_length,
        max_connection_length,
        k_value=2,
        stone_probability=0.5):
    fail_count = 0
    while True:
        # Generate boards with random distribution of stones
        board = np.random.choice(k_value,
                                 size=[shape[0], shape[1]],
                                 p=[1 - stone_probability, stone_probability])

        connection_length = get_connection_length(board)

        if connection_length is None:
            # connectivity = sys.maxsize
            connectivity = -1
        else:
            connectivity = connection_length

        if (min_connection_length <= connectivity <= max_connection_length):
            print ("success after ", fail_count, " failures")
            fail_count = 0  # reset counter
            yield board, connectivity  # X, y
        else:
            fail_count += 1  # print "failed"


def generate_constrained_dataset(
        shape=None,
        num_examples=None,
        shuffle_dataset=True,
        min_lenght=0,
        max_length=None,
        stone_probability=0.5):
        # filename,
        # size,
        # num_positive_examples,
        # num_negative_examples,
        # stone_probability=0.5,
        # k_value=2,
        # verbose=False):
        # progress_fn=None,
    '''
    Generates a constrained dataset of 50/50 positive and negative examples
        and converts them to a .tfrecords file on the disk where filepath
        is specified.

    Args:
        filepath: path to file to be written
        progress_fn: progress logger hook (unused)
        shape: shape of the data to be written
        num_examples: number of examples to generate, half of which will be positive
        shuffle: whether or not to shuffle this dataset before writing to file (True)
        stone_probability: probability distribution of 'black' stones on the generated boards
    '''

    width = shape[0]
    height = shape[1]

    theoretical_max_length = int(height * width / 2 + width / 2)
    inputs = np.empty((num_examples, width, height), np.int8)  # boards
    labels = np.empty((num_examples, ), np.int8)  # connection length

    pos_board_generator = board_generator(
        shape, min_lenght, theoretical_max_length, stone_probability=stone_probability)
    neg_board_generator = board_generator(
        shape, -1, -1, stone_probability=stone_probability)

    num_pos_examples = num_examples // 2

    for i in range(num_examples):
        if i < num_pos_examples:
            inputs[i], labels[i] = next(pos_board_generator)
            # print ("positive board generated.")
        else:
            inputs[i], labels[i] = next(neg_board_generator)

        if i % 1000 == 0:
            print('Generating {} boards. Progress: {}/{}'.format(
                'positive' if i < num_pos_examples else 'negative', i, num_examples))

    if shuffle_dataset:
        inputs, labels = shuffle(inputs, labels, random_state=1234)
        # perm = np.random.permutation(len(inputs))
        # inputs = inputs[perm]
        # labels = labels[perm]

    return inputs, labels


def _bytes_feature(values):
    '''Helper function to generate an tf.train.Feature protobuf'''
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(values):
    '''Helper function to generate an tf.train.Feature protobuf'''
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _data_to_tfexample(encoded_image, height, width, class_id):
    '''Helper function to generate an tf.train.Example protobuf'''
    return tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': _bytes_feature(encoded_image)
            # 'image/class/label': _int64_feature(class_id),
            # 'image/height': _int64_feature(height),
            # 'image/uint8image': _int64_feature(uint8image)
        }))


def _convert_to_tfrecords(inputs, shape, labels, filepath):
    '''Helper function to write tfrecords file

    Args:
        inpupts:
        shape:
        labels:
        filepath:
    '''
    # print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)

    inputs = inputs.astype(np.int8)
    labels = labels.astype(np.int8)

    for i, input_ in enumerate(inputs):
        features = input_.tobytes()
        label = labels[i].tobytes()

        # Encode label byte as part of the record
        encoded_image = b''.join([label, features])

        example = _data_to_tfexample(encoded_image, shape[0], shape[1], label)
        # size = example.BytesSize()
        writer.write(example.SerializeToString())
    writer.close()


# Define record file reader
def read_and_decode(filename_queue, shape=None):
    """Read and decode "tfrecord" binary files

    Args:
        filename_queue: a queue of filenames generated from
            tf.train.string_input_producer
        shape: shape of input [width x height x depth]

    Returns:
        example: a training example
        label: its corresponding label
    """
    label_bytes = 1
    width = shape[0]
    height = shape[1]
    depth = shape[2]
    record_byte_length = label_bytes + width * height

    with tf.name_scope("read_and_decode"):
        # Length of record bytes in the dataset
        # Defined in utils module
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)

        feature_map = {
            "image/encoded": tf.FixedLenFeature(
                shape=[], dtype=tf.string)
        }
        parsed = tf.parse_single_example(record_string, feature_map)
        record_bytes = tf.decode_raw(parsed["image/encoded"], tf.int8)

        # first byte is the label
        label = tf.cast(tf.strided_slice(record_bytes,
                                         begin=[0],
                                         end=[label_bytes]), tf.int32)
        # label = tf.reshape(label, [1])
        # print(label)

        # remaining bytes is the example
        example = tf.reshape(tf.strided_slice(record_bytes,
                                              begin=[label_bytes],
                                              end=[record_byte_length]), [width, height, depth])
        example = tf.cast(example, tf.float32)
        example.set_shape([width, height, depth])
        label.set_shape(1)
        label = tf.squeeze(label)
        # print(label)
        # label = tf.reshape(label, [0])

    return example, label


def generate_batch(examples, labels, batch_size, num_threads, istrain):
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES *
                             min_fraction_of_examples_in_queue)

    if istrain:
        example_batch, label_batch = tf.train.shuffle_batch(
            [examples, labels],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_threads,
            min_after_dequeue=min_queue_examples)
    else:
        example_batch, label_batch = tf.train.batch(
            [examples, labels],
            batch_size=batch_size,
            capacity=min_queue_examples + 3 * batch_size,
            num_threads=num_threads)
    return example_batch, label_batch


# Create input pipeline from reader
def input_pipeline(data_dir, batch_size, shape=None, num_threads=1, num_files=1, istrain=True, name=None):
    with tf.name_scope(name or "input_pipeline_batches"):
        data_dir = os.path.join(data_dir, "batches-bin")
        if istrain:
            filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i)
                         for i in range(num_files)]
        else:
            filenames = [os.path.join(data_dir, "test_batch_%d.bin" % i)
                         for i in range(num_files)]

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError("Failed to find file: " + f)

        filename_queue = tf.train.string_input_producer(filenames)
        examples, labels = read_and_decode(filename_queue, shape)
    return generate_batch(examples, labels, batch_size, num_threads, istrain)


def embedding_metadata(data_dir, shape, n):
    """Extract a specific number of items from binary file

    Args:
        data_dir:
        shape:
        n: number of inputs to extract

    Returns:
        inputs, labels
    """
    with tf.Session() as sess:
        imgs, lbs = input_pipeline(data_dir, n, shape, istrain=False)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            im, lb = sess.run([imgs, lbs])
        finally:
            coord.request_stop()
            coord.join(threads)
    return im, lb


def save_emdedding_metadata(data_dir, shape, n):
    """Save dataset metadata to file using csv and matplotlib modules"""
    im, lb = embedding_metadata(data_dir, shape, n)

    # Generate tab separated file
    with open("labels_" + str(n) + ".tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for i, l in enumerate(lb):
            writer.writerow(str(l))

    # Generate sprite image: an image containing n
    #   number of sub images tiled
    import matplotlib.image as image

    imgs = np.squeeze(im)
    rows = []
    num_rows = int(np.sqrt(n))
    num_cols = int(np.sqrt(n))

    def index(i, j):
        return i + j * num_cols

    for j in range(num_rows):
        row = imgs[j * num_rows]
        for i in range(num_cols - 1):
            row = np.concatenate((row, imgs[index(i + 1, j)]), axis=1)
        rows.append(row)

    img = rows[0]
    for i in range(num_rows - 1):
        img = np.concatenate((img, rows[i + 1]), axis=0)

    # print (img.shape)
    # for i in range(len(im)):
    #     ii = im[i].squeeze()
    #     image.imsave("tmp/sprite" + str(i) + ".png", ii, cmap=image.cm.binary)

    image.imsave("sprite_" + str(n) + ".png", img, cmap=image.cm.binary)


def maybe_generate_data(data_dir,
                        shape=None,
                        num_examples=None,
                        stone_probability=0.45,
                        num_files=2):
    """Generate testing and training data if none exists in data_dir"""
    dest_dir = os.path.join(data_dir, "batches-bin")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Log hook to measure progress
    # TODO: not in use
    def _progress(count, block_size, total_size):
        sys.stdout.write("\r>> Generating %s %.1f%%" % (filename,
                                                        float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    # generate training batches
    # constrained
    filenames = ["data_batch_%d.bin" % i for i in range(num_files)]
    for filename in filenames:
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print("%s not found - generating..." % filename)
            x, y = generate_constrained_dataset(_progress, **{
                "num_examples": num_examples or NUM_EXAMPLES,
                "stone_probability": stone_probability,
                "shape": shape})
            _convert_to_tfrecords(x, shape, y, filepath)
            print()
            statinfo = os.stat(filepath)
            print("Successfully generated", filename,
                  statinfo.st_size, "bytes.")

    # generate testing batches
    # random
    # TODO: generate random dataset
    filenames = ["test_batch_%d.bin" % i for i in range(num_files)]
    for filename in filenames:
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print("%s not found - generating..." % filename)
            # utils.generate_dataset(filepath, _progress, **{
            x, y = generate_constrained_dataset(_progress, **{
                "num_examples": num_examples or NUM_EXAMPLES,
                "stone_probability": stone_probability,
                "shape": shape})
            _convert_to_tfrecords(x, shape, y, filepath)
            print()
            statinfo = os.stat(filepath)
            print("Successfully generated", filename,
                  statinfo.st_size, "bytes.")
