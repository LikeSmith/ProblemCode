"""
DoubleIntegratorDynamics.py

Basic MLP Dynamics estimation model

params:
dt
use_params
hidden_size
hidden_activ
use_bias
init_var
"""

import tensorflow as tf

from .AbstractDynamics import AbstractDynamics
from .Serialize import activ_deserialize

class DoubleIntegratorDynamics(AbstractDynamics):
    #bn_hidden = None

    def build_network(self, inputs=None, weights=None, scope=None):
        dof = int(self.state_size/2)

        if scope is None:
            scope = self.name
        else:
            scope = scope.name + self.name

        comb_size = self.abs_size+self.state_size
        if 'use_params' in self.params.keys() and self.params['use_params']:
            comb_size += self.param_size

        with tf.variable_scope(scope) as local_scope:
            if inputs is None:
                inputs = {}
                inputs['abst_states'] = tf.placeholder('float', [None, self.abs_size], 'abst_states')
                inputs['prnt_states'] = tf.placeholder('float', [None, self.state_size], 'cur_state')
                inputs['prnt_params'] = tf.placeholder('float', [None, self.param_size], 'sta_param')
                inputs['training'] = tf.placeholder_with_default(False, None, 'training')

                self.dt = tf.constant(self.params['dt'], name='dt')

                self.local_scope = local_scope
            else:
                assert 'abst_states' in inputs.keys() and 'prnt_states' in inputs.keys() and 'training' in inputs.keys()

            if weights is None:
                try:
                    weights = self.weights
                except AttributeError:
                    w_init = tf.random_normal_initializer(0, self.params['init_var'])
                    weights = {}
                    weights['W1'] = tf.get_variable('W1', [comb_size, self.params['hidden_size']], initializer=w_init)
                    weights['W2'] = tf.get_variable('W2', [self.params['hidden_size'], dof], initializer=w_init)
                    if self.params['use_bias']:
                        weights['b1'] = tf.get_variable('b1', self.params['hidden_size'], initializer=w_init)
                        weights['b2'] = tf.get_variable('b2', dof, initializer=w_init)
            '''if self.bn_hidden is None:
                self.bn_hidden = tf.layers.BatchNormalization(name='hidden_bn')'''

            outputs = {}
            outputs['prnt_states'] = self.build_dynamics(inputs['abst_states'], inputs['prnt_states'], inputs['prnt_params'], weights=weights, training=inputs['training'])

            return inputs, outputs, weights, {}, {}

    def build_dynamics(self, s_a, s_p, p_p, weights=None, training=False):
        if weights is None:
            weights = self.weights

        dof = int(self.state_size/2)
        p = s_p[:, 0:dof]
        v = s_p[:, dof:]

        if 'use_params' in self.params.keys() and self.params['use_params']:
            comb = tf.concat((s_a, s_p, p_p), axis=1)
        else:
            comb = tf.concat((s_a, s_p), axis=1)

        h = tf.matmul(comb, weights['W1'])
        if self.params['use_bias']:
            h = tf.add(h, weights['b1'])
        #h = self.bn_hidden(h, training=training)
        h = activ_deserialize(self.params['hidden_activ'])(h)

        a_new = tf.matmul(h, weights['W2'])
        if self.params['use_bias']:
            a_new = tf.add(a_new, weights['b2'])

        v_new = v + self.dt*a_new
        p_new = p + self.dt*v_new

        return tf.concat((p_new, v_new), axis=1)
