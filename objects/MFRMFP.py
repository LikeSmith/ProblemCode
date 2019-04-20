"""
MFRMFP.py

Models for the MFRMFP

Plane params:
del_t
p_min
p_max
v_min
v_max
theta_min
theta_max
omega_min
omega_max
clip_p
clip_v
clip_theta
clip_omega

SI Swarm params:
del_t
p_min
p_max
v_min
v_max
clip_p
clip_v

DI Swarm params:
del_t
p_min
p_max
v_min
v_max
f_min
f_max
clip_p
clip_v
clip_f
"""

import tensorflow as tf
from .Dynamics import Dynamics

class MFRMFP_Plane(Dynamics):
    def __init__(self, params={}, load_file=None, home='', name='MFRMFP_Plane', sess=None, debug=False):
        super(MFRMFP_Plane, self).__init__((2,), (None, 3), (8,), params, load_file, home, name, sess, debug)

    def setup_weights(self):
        weights = {}
        weights['del_t'] = tf.constant(self.params['del_t'], name='del_t')

        return weights

    def build_dynamics(self, s, a, par, k, weights=None):
        if weights is None:
            weights = self.weights

        theta = s[:, 0]
        omega = s[:, 1]

        J = par[:, 0]
        gamma_1 = par[:, 1]
        gamma_2 = par[:, 2]
        gamma_3 = gamma_2 - par[:, 3]
        gamma_4 = gamma_1 + par[:, 4]
        gamma_5 = par[:, 5]
        gamma_6 = par[:, 6]
        g = par[:, 7]

        p = a[:, :, 0]
        v = a[:, :, 1]
        m = a[:, :, 2]

        if self.params['clip_p']:
            p = tf.clip_by_value(p, self.params['p_min'], self.params['p_max'])
        if self.params['clip_v']:
            v = tf.clip_by_value(v, self.params['v_min'], self.params['v_max'])

        f_stribeck = gamma_1*(tf.tanh(gamma_2*omega) - tf.tanh(gamma_3*omega)) + gamma_4*tf.tanh(gamma_5*omega) + gamma_6*omega

        tau_s = g*tf.reduce_sum(tf.multiply(p, m), axis=1)
        J_s = tf.reduce_sum(tf.multiply(p**2, m), axis=1)
        J_s_dot = tf.reduce_sum(tf.multiply(2*p*v, m), axis=1)

        alpha_ = -(tf.cos(theta)*tau_s + omega*J_s_dot + f_stribeck)/(J + J_s)
        omega_ = omega + weights['del_t']*alpha_
        if self.params['clip_omega']:
            omega_ = tf.clip_by_value(omega_, self.params['omega_min'], self.params['omega_max'])
        theta_ = theta + weights['del_t']*omega_
        if self.params['clip_theta']:
            theta_ = tf.clip_by_value(theta_, self.params['theta_min'], self.params['theta_max'])

        s_ = tf.stack([theta_, omega_], axis=1)
        return s_

class MFRMFP_Swarm_SI(Dynamics):
    def __init__(self, params={}, load_file=None, home='', name='MFRMFP_Swarm_SI', sess=None, debug=False):
        super(MFRMFP_Swarm_SI, self).__init__((None, 2,), (None, 1), (None, 1), params, load_file, home, name, sess, debug)

    def setup_weights(self):
        weights = {}
        weights['del_t'] = tf.constant(self.params['del_t'], name='del_t')

        return weights

    def build_dynamics(self, s, a, par, k, weights=None):
        if weights is None:
            weights = self.weights

        p = s[:, :, 0]
        v_ = a[:, :, 0]

        if self.params['clip_v']:
            v_ = tf.clip_by_value(v_, self.params['v_min'], self.params['v_max'])
        p_ = p + weights['del_t']*v_
        if self.params['clip_p']:
            p_ = tf.clip_by_value(p_, self.params['p_min'], self.params['p_max'])

        s_ = tf.stack([p_, v_], axis=2)
        return s_
