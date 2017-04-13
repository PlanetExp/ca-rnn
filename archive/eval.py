from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from datetime import datetime
import time
import math

from model import ConvolutionalCA


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'tmp/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, eval_op, summary_op, *args):
    '''

    '''
    # l = tf.cast(args[1], tf.int64)
    # pred2 = tf.equal(tf.argmax(args[2], 1), l)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            # take global_step from the checkpoint file path
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print ('No checkpoint file found')
            return

        # Start runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))  # ?

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(eval_op)
                # label, board, preds, log = sess.run([args[1], args[0], pred2, args[2]])
                
                true_count += np.sum(predictions)
                step += 1
            # print('logits\n', log[0:5])
            # print('true count: ', true_count)
            # print('total sample count: ', total_sample_count)
            # print('preds ', preds[0:5], '\nanswers:', label[0:5])
            # print('for boards:\n', board[0:5, :, 0, 0])
            # print('prediction mean: ', predictions)

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():

  with tf.Graph().as_default() as g:

    # Select which model to evaluate
    Model = ConvolutionalCA()

    # Get boards and labels
    eval_data = FLAGS.eval_data == 'test'
    inputs, labels = Model.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = Model.inference(inputs)

    # Calculate predictions.
    eval_op = Model.prediction(logits, labels)

    # Restore the moving average version of the learned variables for eval.
    # variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    # variables_to_restore = variable_averages.variables_to_restore()
    # saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
        eval_once(saver, summary_writer, eval_op, summary_op, *[inputs, labels, logits])
        if FLAGS.run_once:
            break
        time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
