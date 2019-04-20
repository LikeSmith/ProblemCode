"""
AbstractEncoder.py

Base class for Abstract encoder networks
"""

import tensorflow as tf

from .Model import Model

class AbstractEncoder(Model):
    def __init__(self, abs_size, state_size, param_size, params={}, load_file=None, home='', name='base_model', sess=None, debug=False):
        self.abs_size = abs_size
        self.state_size = state_size
        self.param_size = param_size

        super(AbstractEncoder, self).__init__(params=params, load_file=load_file, home=home, name=name, sess=sess, debug=debug)

    def disect_swarm(self, swarm_states, swarm_params):
        return [swarm_states], [swarm_params]

    def predict(self, inputs):
        return self.sess.run(self.output_ops['abst_states'], feed_dict={self.inputs['swrm_states']:inputs['swrm_states'], self.inputs['swrm_params']:inputs['swrm_params']})

    def encode(self, swrm_states, swrm_params):
        return self.sess.run(self.output_ops['abst_states'], feed_dict={self.inputs['swrm_states']:swrm_states, self.inputs['swrm_params']:swrm_params})
