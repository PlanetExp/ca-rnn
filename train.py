import tensorflow as tf

from datetime import datetime
import time

import model


# Basic parameters.
FLAGS = tf.app.flags.FLAGS  # tensorflows internal argparse
tf.app.flags.DEFINE_string('train_dir', 'tmp/train',
                           '''Directory where to write event logs.'''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            '''Whether to log device placement.''')
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            '''How often to log results to the console.''')


def run_training():
    '''
    version 0.2
        + separates training and eval
        + uses new monitored session object
        + switched from argparse to tf.app.flags in order to share flags between files (tf magic)
    '''
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # model = Model(
        #     batch_size=FLAGS.batch_size,
        #     state_size=FLAGS.state_size,
        #     num_classes=FLAGS.num_classes,
        #     rnn_size=FLAGS.rnn_size,
        #     learning_rate=FLAGS.learning_rate,
        #     cell_name=FLAGS.cell_name)

        # Get training inputs
        boards, labels = model.train_inputs()

        # Build Graph that computes logits predictions from inference
        logits = model.inference(boards)

        # Calculate loss from loss function
        loss = model.loss(logits, labels)

        # Build Graph that trains the model one batch at the time
        # and update parameters
        train_op = model.optimizer(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            '''Hook that logs loss and runtime.'''

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # fetches, can be list

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    run_training()


if __name__ == '__main__':
    tf.app.run()  # automatically adds flags
