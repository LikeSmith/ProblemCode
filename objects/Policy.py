"""
Policy.py

Policy Model base class
"""

import tensorflow as tf

from .Model import Model

class Policy(Model):
    def __init__(self, state_size, actin_size, param_size, params={}, load_file=None, home='', name='Policy', sess=None, debug=False):
        self.state_size = state_size
        self.actin_size = actin_size
        self.param_size = param_size

        super(Policy, self).__init__(params=params, load_file=load_file, home=home, name=name, sess=sess, debug=debug)

    def setup_weights(self):
        raise NotImplementedError

    def build_policy(self, s, p, k, weights=None):
        raise NotImplementedError

    def build_loss(self, a, a_dist, adv, weights=None):
        raise NotImplementedError

    def build_network(self, inputs=None, weights=None):
        with tf.variable_scope(self.name):
            if inputs is None:
                inputs = {}

                inputs['s'] = tf.placeholder('float', (None,) + self.state_size, 's0')
                inputs['p'] = tf.placeholder('float', (None,) + self.param_size, 'p')
                inputs['k'] = tf.placeholder('int32', (), 'k')

            assert 's' in inputs.keys()
            assert 'p' in inputs.keys()
            assert 'k' in inputs.keys()

            if weights is None:
                try:
                    weights = self.weights
                except AttributeError:
                    weights = self.setup_weights()

            a = self.build_policy(inputs['s'], inputs['p'], inputs['k'], weights)

            output_ops = {}
            if isinstance(a, tuple):
                output_ops['a_train'] = a[0]
                output_ops['a'] = a[1]
            else:
                output_ops['a'] = a

            return inputs, output_ops, weights, {}, {}

    def __call__(self, s, p, k, training=False):
        self.pol(s, p, k, training)

    def pol(self, s, p, k, training=False):
        if training and 'a_train' in self.output_ops.keys():
            return self.sess.run(self.output_ops['a_train'], feed_dict={self.inputs['s']:s, self.inputs['p']:p, self.inputs['k']:k})
        else:
            return self.sess.run(self.output_ops['a'], feed_dict={self.inputs['s']:s, self.inputs['p']:p, self.inputs['k']:k})


class Policy_AbsTrack(Model):
    def __init__(self, state_size, actin_size, abstr_size, param_size, params={}, load_file=None, home='', name='Policy', sess=None, debug=False):
        self.state_size = state_size
        self.actin_size = actin_size
        self.param_size = param_size
        self.abstr_size = abstr_size

        super(Policy_AbsTrack, self).__init__(params=params, load_file=load_file, home=home, name=name, sess=sess, debug=debug)

    def setup_weights(self):
        raise NotImplementedError

    def build_policy(self, s_swm, s_abs, s_des, p, k, weights=None):
        raise NotImplementedError

    def build_loss(self, a, a_dist, adv, weights=None):
        raise NotImplementedError

    def build_network(self, inputs=None, weights=None):
        with tf.variable_scope(self.name):
            if inputs is None:
                inputs = {}

                inputs['s_swm'] = tf.placeholder('float', (None,) + self.state_size, 's_swm')
                inputs['s_abs'] = tf.placeholder('float', (None, self.abstr_size), 's_swm')
                if 'n_ders' in self.params.keys() and self.params['n_ders'] > 0:
                    inputs['s_des'] = tf.placeholder('float', (None, self.params['n_ders']+1, self.abstr_size))
                else:
                    inputs['s_des'] = tf.placeholder('float', (None, self.abstr_size))
                inputs['p'] = tf.placeholder('float', (None,) + self.param_size, 'p')
                inputs['k'] = tf.placeholder('int32', (), 'k')

            assert 's_swm' in inputs.keys()
            assert 's_abs' in inputs.keys()
            assert 's_des' in inputs.keys()
            assert 'p' in inputs.keys()
            assert 'k' in inputs.keys()

            if weights is None:
                try:
                    weights = self.weights
                except AttributeError:
                    weights = self.setup_weights()

            a = self.build_policy(inputs['s_swm'], inputs['s_abs'], inputs['s_des'], inputs['p'], inputs['k'], weights)

            output_ops = {}
            if isinstance(a, tuple):
                output_ops['a_train'] = a[0]
                output_ops['a'] = a[1]
            else:
                output_ops['a'] = a

            return inputs, output_ops, weights, {}, {}

    def __call__(self, s_swm, s_abs, s_des, p, k, training=False):
        self.pol(s_swm, s_abs, s_des, p, k, training)

    def pol(self, s_swm, s_abs, s_des, p, k, training=False):
        if training and 'a_train' in self.output_ops.keys():
            return self.sess.run(self.output_ops['a_train'], feed_dict={self.inputs['s_swm']:s_swm, self.inputs['s_abs']:s_abs, self.inputs['s_des']:s_des, self.inputs['p']:p, self.inputs['k']:k})
        else:
            return self.sess.run(self.output_ops['a'], feed_dict={self.inputs['s_swm']:s_swm, self.inputs['s_abs']:s_abs, self.inputs['s_des']:s_des, self.inputs['p']:p, self.inputs['k']:k})