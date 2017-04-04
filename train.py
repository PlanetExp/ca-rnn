import tensorflow as tf

from datetime import datetime
import time

import model


FLAGS = tf.app.flags.FLAGS
'''Global flags interface for command line arguments

    flag_name, default_value, doc_string

Usage:
    python <file> [--flag_name=value]

Uses argparse module internally
See: https://www.tensorflow.org/api_docs/python/tf/flags
'''
tf.app.flags.DEFINE_string('train_dir', 'tmp/train',
                           '''Directory where to write event logs.'''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('model_name', 'conv',
                           '''Select which model to train.'''
                           '''Either conv or rnn.''')
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            '''Whether to log device placement.''')
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            '''How often to log results to the console.''')


def run_training():
    '''
    version 0.1
        + added model selector
        + added automatic generation of training and testing data if none exists
        + separates training and eval
        + uses new monitored session object
        + switched from argparse to tf.app.flags in order to share flags between files (tf magic)
    '''
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Select which model class to train.
        if FLAGS.model_name == 'conv':
            model_fn = model.CAConv
        elif FLAGS.model_name == 'rnn':
            model_fn = model.CARNN
        else:
            raise Exception('Unsupported model_name: {}'.format(FLAGS.model_name))
        mdl = model_fn()

        # Get training inputs
        boards, labels = mdl.train_inputs()

        # Build Graph that computes logits predictions from inference
        logits = mdl.inference(boards)

        # Calculate loss from loss function
        loss = mdl.loss(logits, labels)

        # Build Graph that trains the model one batch at the time
        # and update parameters
        train_op = mdl.optimizer(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            '''Hook that logs loss and runtime.'''

            def begin(self):
                '''Step to run before run starts'''
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                '''Step to run before every fetch'''
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # fetches, can be list

            def after_run(self, run_context, run_values):
                '''Step tun run after every fetch'''
                # Output logs at log_frequency steps
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

        # Start a monitored training session that runs above hook function at every step
        # and manages queue runners and checkpoints automatically.
        # See: 

        # debug = 1
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                # if debug % 100 == 0:
                    # log, preds, corr = mon_sess.run([logits, pred_op, correct])
                    # print('logits\n', log[0:5], 'pred:', preds, 'corr:\n', corr[0:5])
                # debug += 1

                mon_sess.run(train_op)


def main(argv=None):
    model.maybe_generate_data()
    # flush train_dir
    # remove this to start training where train_op previously left off
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    run_training()


if __name__ == '__main__':
    # Run program with optional 'main' function and 'argv' list.
    # main() by default
    tf.app.run()
