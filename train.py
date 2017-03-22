import tensorflow as tf

import argparse
import os.path
import sys
import time

from model import Model

# Basic model parameters as external flags.
FLAGS = None


def run_training():

    model = Model(
        batch_size=FLAGS.batch_size,
        state_size=FLAGS.state_size,
        num_classes=FLAGS.num_classes,
        rnn_size=FLAGS.rnn_size,
        learning_rate=FLAGS.learning_rate,
        cell_name=FLAGS.cell_name)

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()  # init local variable epochs if used

    num_examples = 0
    with tf.Session() as sess:
        sess.run([init, init_local])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # writer = tf.summary.FileWriter('./graphs/run1', sess.graph)

        try:
            step = 0
            total_loss = .0
            print('{:#^40}'.format(' Training ' + FLAGS.cell_name + ' '))
            print()
            while not coord.should_stop():
                step += 1

                loss, _ = sess.run([model.loss, model.optimizer])
                total_loss += loss

                num_examples = num_examples + FLAGS.batch_size
                if step % 100 == 0:
                    print('steps: {steps} num_examples: {num_examples:,} average_loss: {avg_loss:.4}'.format(avg_loss=(total_loss / step), steps=step, num_examples=num_examples))

                if step == FLAGS.max_steps:  # stop early
                    coord.request_stop()

        except tf.errors.OutOfRangeError:
            # print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            print('Requesting to stop')
            coord.request_stop()
        finally:
            coord.request_stop()  # ask threads to to stop
            coord.join(threads)  # wait for threads to finish

            print('{:#^40}'.format(' Finished '))

        # writer.close()


def main(argv=None):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--state_size',
        type=int,
        default=8,
        help='State size of cells.'
    )
    parser.add_argument(
        '--cell_name',
        type=str,
        default='lstm',
        help='Name of cell function to use.'
    )
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=5,
        help='Size of the rnn.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Number of target classes.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
