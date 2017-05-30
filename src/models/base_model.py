
import json
import copy
import os
import tensorflow as tf


class BaseModel(object):
    def __init__(self, config):

        if config['best']:
            config.update(self.get_best_config(config['env_name']))

        self.config = copy.deepcopy(config)

        if config['debug']:
            print('config: ', self.config)

        self.random_seed = self.config['random_seed']

        # Shared basic hyper parameters
        self.result_dir = self.config['result_dir']
        self.max_steps = self.config['max_steps']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.lrelu_rate = self.config['lrelu_rate']
        self.epsilon = self.config['epsilon']
        self.data_dir = self.config['data_dir']

        # Model specific parameters
        self.set_properties()

        self.graph = self.build_graph(tf.Graph())

        # Operations that should be in the graph but are common to all models
        with self.graph.as_default():
            self.saver = tf.train.Saver()

        # Add all the other common code for the initialization here
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(self.result_dir, self.sess.graph)

        self.setup()

    def set_properties(self):
        pass

    def optimize(self):
        raise Exception('Optimize must be implemented by derived model')


    def train(self, save_frequency=-1):
        try:
            for epoch_id in range(0, self.max_steps):
                self.optimize()

                # If you don't want to save during training, you can just pass a negative number
                if save_frequency > 0 and epoch_id % save_frequency == 0:
                    self.save()
        finally:
            self.sess.close()
            self.writer.close()

    def setup(self):
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, epoch_id = self.sess.run([global_step_t, self.epoch_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(epoch_id), global_step)

        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def get_best_config(self):
        raise Exception('get_best_config must be implemented by derived model')
        # return {} 

    def build_graph(self, graph):
        """Build graph and return logits"""
        raise Exception('Inference must be implemented by derived model')
