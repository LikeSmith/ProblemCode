"""
HierarchicalEncoder.py

Hierarchical method of encoding abstract state

params:
max_swarm_size
a_limit
init_var
tran_h_size
tran_use_bias
tran_activ
tran_dropout_rate
comb_h_size
comb_use_bias
comb_activ
comb_dropout_rate
subsample
"""

import tensorflow as tf

from .AbstractEncoder import AbstractEncoder
from .Serialize import activ_deserialize

class HierarchicalEncoder(AbstractEncoder):
    '''bn_hidden_tran = None
    bn_output_tran = None
    bn_hidden_comb = None
    bn_output_comb = None'''
    droput_tran = None
    droput_comb = None

    def build_network(self, inputs=None, weights=None, scope=None):
        if scope is None:
            scope = self.name
        else:
            scope = scope.name + self.name

        with tf.variable_scope(scope) as local_scope:
            if inputs is None:
                inputs = {}
                inputs['swrm_states'] = tf.placeholder('float', [None, self.params['max_swarm_size'], self.state_size], 'swrm_s')
                inputs['swrm_params'] = tf.placeholder('float', [None, self.params['max_swarm_size'], self.param_size], 'swrm_p')
                inputs['swrm_state'] = tf.placeholder('float', [None, self.state_size], 's_s')
                inputs['swrm_param'] = tf.placeholder('float', [None, self.param_size], 'p_s')
                inputs['abst_state1'] = tf.placeholder('float', [None, self.abs_size], 'a_1')
                inputs['abst_state2'] = tf.placeholder('float', [None, self.abs_size], 'a_2')
                inputs['training'] = tf.placeholder_with_default(False, None, 'training')

                if 'a_limit' in self.params.keys():
                    self.a_limit = tf.constant(self.params['a_limit'], name='a_limit')

                self.local_scope = local_scope

            if weights is None:
                try:
                    weights = self.weights
                except AttributeError:
                    w_init = tf.random_normal_initializer(0, self.params['init_var'])
                    weights = {}
                    weights['tran_W1'] = tf.get_variable('tran_W1', [self.state_size+self.param_size, self.params['tran_h_size']], initializer=w_init)
                    weights['tran_W2'] = tf.get_variable('tran_W2', [self.params['tran_h_size'], self.abs_size], initializer=w_init)
                    if self.params['tran_use_bias']:
                        weights['tran_b1'] = tf.get_variable('tran_b1', self.params['tran_h_size'], initializer=w_init)
                        weights['tran_b2'] = tf.get_variable('tran_b2', self.abs_size, initializer=w_init)
                    weights['comb_W1'] = tf.get_variable('comb_W1', [2*self.abs_size, self.params['comb_h_size']], initializer=w_init)
                    weights['comb_W2'] = tf.get_variable('comb_W2', [self.params['comb_h_size'], self.abs_size], initializer=w_init)
                    if self.params['comb_use_bias']:
                        weights['comb_b1'] = tf.get_variable('comb_b1', self.params['comb_h_size'], initializer=w_init)
                        weights['comb_b2'] = tf.get_variable('comb_b2', self.abs_size, initializer=w_init)

            '''if self.bn_hidden_tran is None:
                self.bn_hidden_tran = tf.layers.BatchNormalization(name='tran_hidden_bn')
                self.bn_output_tran = tf.layers.BatchNormalization(name='tran_output_bn')
                self.bn_hidden_comb = tf.layers.BatchNormalization(name='comb_hidden_bn')
                self.bn_output_comb = tf.layers.BatchNormalization(name='comb_output_bn')'''
            if self.droput_tran is None and 'tran_dropout_rate' in self.params.keys():
                self.dropout_tran = tf.keras.layers.Dropout(rate=self.params['tran_dropout_rate'])
            if self.droput_comb is None and 'comb_dropout_rate' in self.params.keys():
                self.dropout_comb = tf.keras.layers.Dropout(rate=self.params['comb_dropout_rate'])

            outputs = {}

            if 'swrm_state' in inputs.keys() and 'swrm_param' in inputs.keys():
                outputs['abst_state'] = self.build_tran(inputs['swrm_state'], inputs['swrm_param'], inputs['training'], weights)

            if 'abst_state1' in inputs.keys() and 'abst_state2' in inputs.keys():
                outputs['comb_state'] = self.build_comb(inputs['abst_state1'], inputs['abst_state2'], inputs['training'], weights)

            if 'swrm_states' in inputs.keys() and 'swrm_params' in inputs.keys():
                abs_states = []
                abs_states.append([])
                constituents = []
                constituents.append([])
                for i in range(self.params['max_swarm_size']):
                    abs_states[0].append(self.build_tran(inputs['swrm_states'][:, i, :], inputs['swrm_params'][:, i, :], inputs['training'], weights))
                    constituents[0].append([i])

                abs_states, constituents = self.hierarch_build(abs_states, constituents, weights, inputs['training'])

                if 'subsample' in self.params.keys() and self.params['subsample']:
                    outputs['abst_states'] = []
                    const_list = []
                    for i in range(len(abs_states)):
                        for j in range(len(abs_states[i])):
                            outputs['abst_states'].append(abs_states[i][j])
                            const_list.append(constituents[i][j])
                else:
                    outputs['abst_states'] = [abs_states[-1][0]]
                    const_list = [constituents[-1][0]]

                self.constituents = const_list

        return inputs, outputs, weights, {}, {}

    def build_tran(self, swrm_state, swrm_param, training, weights):
        comb_in = tf.concat((swrm_state, swrm_param), axis=1)
        h = tf.matmul(comb_in, weights['tran_W1'])
        if self.params['tran_use_bias']:
            h = tf.add(h, weights['tran_b1'])
        #h = self.bn_hidden_tran(h, training=training)
        h = activ_deserialize(self.params['tran_activ'])(h)
        if 'tran_dropout_rate' in self.params.keys():
            h = self.dropout_tran(h, training=training)
        abst_state = tf.matmul(h, weights['tran_W2'])
        if self.params['tran_use_bias']:
            abst_state = tf.add(abst_state, weights['tran_b2'])
        if 'a_limit' in self.params.keys():
            #abst_state = self.bn_output_tran(abst_state, training=training)
            abst_state = tf.tanh(abst_state)*self.a_limit
        return abst_state

    def build_comb(self, abst_state_1, abst_state_2, training, weights):
        comb_in = tf.concat((abst_state_1, abst_state_2), axis=1)
        h = tf.matmul(comb_in, weights['comb_W1'])
        if self.params['comb_use_bias']:
            h = tf.add(h, weights['comb_b1'])
        #h = self.bn_hidden_comb(h, training=training)
        h = activ_deserialize(self.params['comb_activ'])(h)
        if 'comb_dropout_rate' in self.params.keys():
            h = self.dropout_comb(h, training=training)
        abst_state = tf.matmul(h, weights['comb_W2'])
        if self.params['comb_use_bias']:
            abst_state = tf.add(abst_state, weights['comb_b2'])
        if 'a_limit' in self.params.keys():
            #abst_state = self.bn_output_comb(abst_state, training=training)
            abst_state = tf.tanh(abst_state)*self.a_limit
        return abst_state

    def build_abs(self, swrm_states, swrm_params, training=False, swarm_size=None, full_state=False):
        abs_states = [[]]
        constituents = [[]]
        if swarm_size is None:
            swarm_size = self.params['max_swarm_size']

        for i in range(swarm_size):
            abs_states[0].append(self.build_tran(swrm_states[:, i, :], swrm_params[:, i, :], training, self.weights))
            constituents[0].append(i)

        abs_states, constituents = self.hierarch_build(abs_states, constituents, self.weights, training)
        if full_state:
            ret = []
            for i in range(len(abs_states)):
                for j in range(len(abs_states[i])):
                    ret.append(abs_states[i][j])
            return ret
        else:
            return abs_states[-1][0]

    def hierarch_build(self, abs_states, constituents, weights, training):
        grp_1 = abs_states[-1][0::2]
        grp_2 = abs_states[-1][1::2]
        grp_1_const = constituents[-1][0::2]
        grp_2_const = constituents[-1][1::2]

        assert len(grp_1) == len(grp_1_const) and len(grp_2) == len(grp_2_const)

        abs_states.append([])
        constituents.append([])

        for i in range(len(grp_2)):
            abs_states[-1].append(self.build_comb(grp_1[i], grp_2[i], training, weights))
            constituents[-1].append(grp_1_const[i]+grp_2_const[i])

        if len(grp_1) != len(grp_2):
            abs_states[-1].append(grp_1[-1])
            constituents[-1].append(grp_1_const[-1])

        if len(abs_states[-1]) == 1:
            return abs_states, constituents
        else:
            return self.hierarch_build(abs_states, constituents, weights, training)

    def predict(self, inputs):
        swarm_size = inputs['swrm_states'].shape[1]

        abs_states = []

        for i in range(swarm_size):
            abs_states.append(self.sess.run(self.output_ops['abst_state'], feed_dict={self.inputs['swrm_state']:inputs['swrm_states'][:, i, :], self.inputs['swrm_param']:inputs['swrm_params'][:, i, :]}))

        return self.combine(abs_states)

    def encode(self, swrm_states, swrm_params, training=False):
        swarm_size = swrm_states.shape[1]

        abs_states = []

        for i in range(swarm_size):
            abs_states.append(self.sess.run(self.output_ops['abst_state'], feed_dict={self.inputs['swrm_state']:swrm_states[:, i, :], self.inputs['swrm_param']:swrm_params[:, i, :], self.inputs['training']:training}))

        return self.combine(abs_states)[0]

    def combine(self, abs_states, training=False):
        grp_1 = abs_states[0::2]
        grp_2 = abs_states[1::2]

        abs_states = []

        for i in range(len(grp_2)):
            abs_states.append(self.sess.run(self.output_ops['comb_state'], feed_dict={self.inputs['abst_state1']:grp_1[i], self.inputs['abst_state2']:grp_2[i], self.inputs['training']:training}))

        if len(grp_1) != len(grp_2):
            abs_states.append(grp_1[-1])

        if len(abs_states) == 1:
            return abs_states
        else:
            return self.combine(abs_states)

    def get_constituents_list(self):
        return self.constituents
