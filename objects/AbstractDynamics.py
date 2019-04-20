"""
AbstractDynamics.py

Base class for Abstract Dynamics network
"""

from .Model import Model

class AbstractDynamics(Model):
    def __init__(self, abs_size, state_size, param_size=0, params={}, load_file=None, home='', name='base_model', sess=None, debug=False):
        self.abs_size = abs_size
        self.state_size = state_size
        self.param_size = param_size

        super(AbstractDynamics, self).__init__(params=params, load_file=load_file, home=home, name=name, sess=sess, debug=debug)

    def predict(self, inputs):
        feed = {}
        for key, val in self.inputs.items():
            feed[val] = inputs[key]

        return self.sess.run(self.output_ops['prnt_states'], feed_dict=feed)

    def step(self, prnt_states, abst_states, prnt_params=None):
        return self.sess.run(self.output_ops['prnt_states'], feed_dict={self.inputs['prnt_states']:prnt_states, self.inputs['abst_states']:abst_states, self.inputs['prnt_params']:prnt_params})