"""
Basic.py

Basic MLP Policy

params:
h_size
use_bias
activ
a_min
a_max
init_var

BasicPol_Swarm_AbsTrack
h_size
use_bias
activ
a_min
a_max
n_ders
individualized
parameterized
max_swarm_size
init_var
"""

import tensorflow as tf

from .Policy import Policy, Policy_AbsTrack
from .Serialize import activ_deserialize

class BasicPol(Policy):
    '''bn_hidden = None
    bn_output = None'''

    def setup_weights(self):
        state_size = self.state_size[0]
        actin_size = self.actin_size[0]
        w_init = tf.random_normal_initializer(0, self.params['init_var'])
        weights = {}
        weights['W1'] = tf.get_variable('W1', (state_size, self.params['h_size']), initializer=w_init)
        weights['W2'] = tf.get_variable('W2', (self.params['h_size'], actin_size), initializer=w_init)
        if self.params['use_bias']:
            weights['b1'] = tf.get_variable('b1', (self.params['h_size']), initializer=w_init)
            weights['b2'] = tf.get_variable('b2', (actin_size), initializer=w_init)

        '''if self.bn_hidden is None:
            self.bn_hidden = tf.layers.BatchNormalization(name='hidden_bn')
            self.bn_output = tf.layers.BatchNormalization(name='output_bn')'''

        return weights

    def build_policy(self, s, p, k, weights=None, training=False):
        if weights is None:
            weights = self.weights

        if 'a_min' in self.params.keys() and 'a_max' in self.params.keys():
            limited = True
            offset = []
            scale = []
            for i in range(self.actin_size[0]):
                offset.append(tf.constant((self.params['a_max'][i] + self.params['a_min'][i])/2.0, name='a_offset%d'%(i,)))
                scale.append(tf.constant((self.params['a_max'][i] - self.params['a_min'][i])/2.0, name='a_scale%d'%(i,)))
        else:
            limited = False

        h = tf.matmul(s, weights['W1'])
        if self.params['use_bias']:
            h = tf.add(h, weights['b1'])
        #h = self.bn_hidden(h, training=training)
        h = activ_deserialize(self.params['activ'])(h)

        a = tf.matmul(h, weights['W2'])
        if self.params['use_bias']:
            a = tf.add(a, weights['b2'])

        if limited:
            #a = self.bn_output(a, training=training)
            a = tf.nn.tanh(a)
            a_sep = []
            for i in range(self.actin_size[0]):
                a_sep.append(a[:, i]*scale[i] + offset[i])
            a = tf.stack(a_sep, axis=1, name='limited_a')

        return a

class BasicPol_Swarm_AbsTrack(Policy_AbsTrack):
    '''bn_hidden = None
    bn_output = None'''

    def setup_weights(self):
        max_swarm_size = self.params['max_swarm_size']
        state_size = self.state_size[1]
        actin_size = self.actin_size[1]
        param_size = self.param_size[1]
        abstr_size = self.abstr_size
        n_ders = 0

        if 'n_ders' in self.params.keys():
            n_ders = self.params['n_ders']

        w_init = tf.random_normal_initializer(0, self.params['init_var'])
        weights = {}

        if 'individualized' in self.params.keys() and self.params['individualized']:
            for i in range(max_swarm_size):
                weights['W1_%d'%i] = tf.get_variable('W1_%d'%i, (state_size+abstr_size*(1+n_ders), self.params['h_size']), initializer=w_init)
                weights['W2_%d'%i] = tf.get_variable('W2_%d'%i, (self.params['h_size'], actin_size), initializer=w_init)
                if self.params['use_bias']:
                    weights['b1_%d'%i] = tf.get_variable('b1_%d'%i, self.params['h_size'], initializer=w_init)
                    weights['b2_%d'%i] = tf.get_variable('b2_%d'%i, actin_size, initializer=w_init)
        elif 'parameterized' in self.params.keys() and self.params['parameterized']:
            weights['W1'] = tf.get_variable('W1', (state_size+param_size+abstr_size*(1+n_ders), self.params['h_size']), initializer=w_init)
            weights['W2'] = tf.get_variable('W2', (self.params['h_size'], actin_size), initializer=w_init)
            if self.params['use_bias']:
                weights['b1'] = tf.get_variable('b1', self.params['h_size'], initializer=w_init)
                weights['b2'] = tf.get_variable('b2', actin_size, initializer=w_init)
        else:
            weights['W1'] = tf.get_variable('W1', (state_size+abstr_size*(1+n_ders), self.params['h_size']), initializer=w_init)
            weights['W2'] = tf.get_variable('W2', (self.params['h_size'], actin_size), initializer=w_init)
            if self.params['use_bias']:
                weights['b1'] = tf.get_variable('b1', self.params['h_size'], initializer=w_init)
                weights['b2'] = tf.get_variable('b2', actin_size, initializer=w_init)

        '''if self.bn_hidden is None:
            self.bn_hidden = tf.layers.BatchNormalization(name='hidden_bn')
            self.bn_output = tf.layers.BatchNormalization(name='output_bn')'''

        return weights

    def build_policy(self, s_swm, s_abs, s_des, p, k, weights=None, members=None, training=False):
        max_swarm_size = self.params['max_swarm_size']
        actin_size = self.actin_size[1]
        n_ders = 0

        if members is None:
            members = range(max_swarm_size)

        if 'n_ders' in self.params.keys():
            n_ders = self.params['n_ders']

        if weights is None:
            weights = self.weights
        if 'a_min' in self.params.keys() and 'a_max' in self.params.keys():
            limited = True
            offset = []
            scale = []
            for i in range(actin_size):
                offset.append(tf.constant((self.params['a_max'][i] + self.params['a_min'][i])/2.0, name='a_offset%d'%(i,)))
                scale.append(tf.constant((self.params['a_max'][i] - self.params['a_min'][i])/2.0, name='a_scale%d'%(i,)))
        else:
            limited = False

        if n_ders != 0:
            s_des_ders = s_des[:, 1:, :]
            s_des = s_des[:, 0, :]

        err = s_des - s_abs

        if n_ders != 0:
            der_stack = []
            for i in range(n_ders):
                der_stack.append(s_des_ders[:, i, :])
            err = tf.concat([err]+der_stack, axis=1)

        if 'individualized' in self.params.keys() and self.params['individualized']:
            a = []
            for i in range(len(members)):
                comb_in = tf.concat((s_swm[:, i, :], err), axis=1)
                h = tf.matmul(comb_in, weights['W1_%d'%members[i]])
                if self.params['use_bias']:
                    h = tf.add(h, weights['b1_%d'%members[i]])
                h = activ_deserialize(self.params['activ'])(h)
                a_i = tf.matmul(h, weights['W2_%d'%members[i]])
                if self.params['use_bias']:
                    a_i = tf.add(a_i, weights['b2_%d'%members[i]])
                if limited:
                    a_i = tf.nn.tanh(a_i)
                    a_sep = []
                    for j in range(actin_size):
                        a_sep.append(a_i[:, j]*scale[j] + offset[j])
                    a_i = tf.stack(a_sep, axis=1, name='limited_a')
                a.append(a_i)
            a = tf.stack(a, axis=1, name='a')
        else:
            if 'parameterized' in self.params.keys() and self.params['parameterized']:
                comb_in = tf.concat((s_swm, p, tf.tile(tf.expand_dims(err, axis=1), (1, tf.shape(s_swm)[1], 1))), axis=2)
            else:
                comb_in = tf.concat((s_swm, tf.tile(tf.expand_dims(err, axis=1), (1, tf.shape(s_swm)[1], 1))), axis=2)
            h = tf.einsum('ijk,kl->ijl', comb_in, weights['W1'])
            if self.params['use_bias']:
                h = tf.add(h, weights['b1'])
            #self.bn_hidden(h, training=training)
            h = activ_deserialize(self.params['activ'])(h)
            a = tf.einsum('ijk,kl->ijl', h, weights['W2'])
            if self.params['use_bias']:
                a = tf.add(a, weights['b2'])
            if limited:
                #a = self.bn_output(a, training=training)
                a = tf.nn.tanh(a)
                a_sep = []
                for i in range(actin_size):
                    a_sep.append(a[:, :, i]*scale[i] + offset[i])
                a = tf.stack(a_sep, axis=2)

        return a
