"""
Dynamics.py

Base class for dynamics model
"""

import tensorflow as tf
from .Model import Model

class Dynamics(Model):
    def __init__(self, state_size, actin_size, param_size, params={}, load_file=None, home='', name='base_model', sess=None, debug=False):
        self.state_size = state_size
        self.actin_size = actin_size
        self.param_size = param_size

        super(Dynamics, self).__init__(params=params, load_file=load_file, home=home, name=name, sess=sess, debug=debug)

    def setup_weights(self):
        raise NotImplementedError

    def build_dynamics(self, s, a, p, k, weights=None):
        raise NotImplementedError

    def build_network(self, inputs=None, weights=None):
        with tf.variable_scope(self.name):
            if inputs is None:
                inputs = {}
                inputs['s'] = tf.placeholder('float', (None,) + self.state_size, 's0')
                inputs['a'] = tf.placeholder('float', (None,) + self.actin_size, 'a')
                inputs['p'] = tf.placeholder('float', (None,) + self.param_size, 'p')
                inputs['k'] = tf.placeholder('float', (), 'k')

            assert 's' in inputs.keys()
            assert 'a' in inputs.keys()
            assert 'p' in inputs.keys()
            assert 'k' in inputs.keys()

            if weights is None:
                try:
                    weights = self.weights
                except AttributeError:
                    weights = self.setup_weights()

            s = self.build_dynamics(inputs['s'], inputs['a'], inputs['p'], inputs['k'], weights)

            outputs = {}
            outputs['s'] = s

            return inputs, outputs, weights, {}, {}

    def __call__(self, s, a, p, k):
        return self.sess.run(self.output_ops['s'], feed_dict={self.inputs['s']:s, self.inputs['a']:a, self.inputs['p']:p, self.inputs['k']:k})

    def step(self, s, a, p, k):
        return self.sess.run(self.output_ops['s'], feed_dict={self.inputs['s']:s, self.inputs['a']:a, self.inputs['p']:p, self.inputs['k']:k})

    def train(self):
        pass
