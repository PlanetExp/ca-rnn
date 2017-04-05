import tensorflow as tf

from datetime import datetime
import time

import model

from model import ConvolutionalCA
from model import RecurrentCA


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


def run_distributed_training():
    '''
    version 0.1
        + added model selector
        + added automatic generation of training and testing data if none exists
        + separates training and eval
        + uses new monitored session object
        + switched from argparse to tf.app.flags in order to share flags between files (tf magic)
    '''
    with tf.Graph().as_default():

        # Select which model class to train.
        if FLAGS.model_name == 'conv':
            model_fn = ConvolutionalCA
        elif FLAGS.model_name == 'rnn':
            model_fn = RecurrentCA
        else:
            raise Exception('Unsupported model_name: {}'.format(FLAGS.model_name))
        Model = model_fn()

        # Build model
        global_step = tf.contrib.framework.get_or_create_global_step()
        inputs, labels = Model.train_inputs()
        logits = Model.inference(inputs)
        loss = Model.loss(logits, labels)
        train_op = Model.optimizer(loss, global_step)

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

        # Start a monitored training session to enable asynchronous training and evaluation
        # See: https://www.tensorflow.org/versions/r1.1/deploy/distributed

        # The StopAtStepHook handles stopping after running given steps.
        # NanTensorHook handles stopping if loss ever happens to be NaN
        # and _loggerHook is our custom logger above
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                 tf.train.NanTensorHook(loss),
                 _LoggerHook()]
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)

        with tf.train.MonitoredTrainingSession(is_chief=True,
                                               checkpoint_dir=FLAGS.train_dir,
                                               hooks=hooks,
                                               config=config) as mon_sess:
            while not mon_sess.should_stop():

                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op)


def run_training():
    Model = ConvolutionalCA()

    # Build model
    global_step = tf.contrib.framework.get_or_create_global_step()
    inputs, labels = Model.train_inputs()
    logits = Model.inference(inputs)
    loss = Model.loss(logits, labels)
    train_op = Model.optimizer(loss, global_step)
    init_op = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init_op)

    # Start a number of parallel input queue threads to feed data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    try:
        start_time = time.time()
        while not coord.should_stop():

            _, loss_value = sess.run([train_op, loss])

            if FLAGS.max_steps == step:
                coord.request_stop()

            if step % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - start_time
                start_time = current_time

                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                           examples_per_sec, sec_per_batch))

            step += 1

    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    model.maybe_generate_data()
    # flush train_dir
    # remove this to start training where train_op previously left off
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    run_distributed_training()
    # run_training()


if __name__ == '__main__':
    # Run program with optional 'main' function and 'argv' list.
    # main() by default
    tf.app.run()
