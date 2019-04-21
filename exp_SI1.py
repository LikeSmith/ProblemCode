"""
exp_SI1.py

generates and saves experimental parameters for single integrator swarm
"""

import numpy as np
import tensorflow as tf

from youngknight.core import rand_vectors

exp_id = 'SI1_exp1'
home = 'results/'

enc_filename = 'abst_end_%s.pkl'%exp_id
dyn_filename = 'abst_dyn_%s.pkl'%exp_id
prt_pol_filename = 'prnt_pol_%s.pkl'%exp_id
swm_pol_filename = 'swrm_pol_%s.pkl'%exp_id

dt = 0.05
t_f = 0.50
verbosity = 2
use_best = True
regen_data = True
patience = 10
lr_min = 10e-6
lr_max = 10
beta = 0.98
lr_count_patience = 10

max_swarm_size = 10
min_swarm_size = 10
prnt_state_size = 2
prnt_param_size = 8
abst_state_size = 4
chld_state_size = 2
chld_actin_size = 1
chld_param_size = 1

gamma_abs = 1.0
gamma_swm = 1.0
gamma_des = 1.0
gamma_bnd = 1.0
Q_p = np.array([[10.0, 0.0], [0.0, 1.0]])
R_i = np.array([[0.1]])

J = 1.5
gamma_1 = 0.01
gamma_2 = 1000
gamma_3 = 700
gamma_4 = 0.07
gamma_5 = 1000
gamma_6 = 0.5
g = 9.81

J_min = J
J_max = J
gamma_1_min = gamma_1
gamma_1_max = gamma_1
gamma_2_min = gamma_2
gamma_2_max = gamma_2
gamma_23_min = gamma_2 - gamma_3
gamma_23_max = gamma_2 - gamma_3
gamma_41_min = gamma_4 - gamma_1
gamma_41_max = gamma_4 - gamma_1
gamma_5_min = gamma_5
gamma_5_max = gamma_5
gamma_6_min = gamma_6
gamma_6_max = gamma_6
g_min = g
g_max = g

theta_min = -np.pi/60.0
theta_max = np.pi/60.0
omega_min = -1.0
omega_max = 1.0

p_min = -2.0
p_max = 2.0
v_min = -1.0
v_max = 1.0
m_min = 1.5
m_max = 2.0

a_limit = 10.0
a_dot_limit = [5.0]

swrm_state_gen = lambda n,s: rand_vectors((n, s), np.array([p_min, v_min]), np.array([p_max, v_max]))
swrm_param_gen = lambda n,s: rand_vectors((n, s), np.array([m_min]), np.array([m_max]))
prnt_state_gen = lambda n: rand_vectors(n, np.array([theta_min, omega_min]), np.array([theta_max, omega_max]))
prnt_param_gen = lambda n: rand_vectors(n, np.array([J_min, gamma_1_min, gamma_2_min, gamma_23_min, gamma_41_min, gamma_5_min, gamma_6_min, g_min]), np.array([J_max, gamma_1_max, gamma_2_max, gamma_23_max, gamma_41_max, gamma_5_max, gamma_6_max, g_max]))

n_steps = int(t_f/dt)+1

full_trainer = tf.train.AdamOptimizer(learning_rate=0.001)

prnt_dyn_params = {}
prnt_dyn_params['del_t'] = dt
prnt_dyn_params['p_min'] = p_min
prnt_dyn_params['p_max'] = p_max
prnt_dyn_params['v_min'] = v_min
prnt_dyn_params['v_max'] = v_max
prnt_dyn_params['theta_min'] = theta_min
prnt_dyn_params['theta_max'] = theta_max
prnt_dyn_params['omega_min'] = omega_min
prnt_dyn_params['omega_max'] = omega_max
prnt_dyn_params['clip_p'] = False
prnt_dyn_params['clip_v'] = False
prnt_dyn_params['clip_theta'] = False
prnt_dyn_params['clip_omega'] = False

swrm_dyn_params = {}
swrm_dyn_params['del_t'] = dt
swrm_dyn_params['p_min'] = p_min
swrm_dyn_params['p_max'] = p_max
swrm_dyn_params['v_min'] = v_min
swrm_dyn_params['v_max'] = v_max
swrm_dyn_params['clip_p'] = False
swrm_dyn_params['clip_v'] = False

abst_dyn_params = {}
abst_dyn_params['hidden_size'] = 200
abst_dyn_params['hidden_activ'] = 'leakyrelu_0.1'
abst_dyn_params['init_var'] = 0.1
abst_dyn_params['dt'] = dt
abst_dyn_params['use_bias'] = False
abst_dyn_params['use_params'] = False

abst_enc_params = {}
abst_enc_params['max_swarm_size'] = max_swarm_size
abst_enc_params['a_limit'] = a_limit
abst_enc_params['init_var'] = 0.1
abst_enc_params['tran_h_size'] = 200
abst_enc_params['tran_use_bias'] = False
abst_enc_params['tran_activ'] = 'leakyrelu_0.1'
abst_enc_params['tran_dropout_rate'] = 0.2
abst_enc_params['comb_h_size'] = 200
abst_enc_params['comb_use_bias'] = False
abst_enc_params['comb_activ'] = 'leakyrelu_0.1'
abst_enc_params['comb_dropout_rate'] = 0.2
abst_enc_params['subsample'] = False

prnt_pol_params = {}
prnt_pol_params['h_size'] = 100
prnt_pol_params['use_bias'] = False
prnt_pol_params['activ'] = 'leakyrelu_0.1'
prnt_pol_params['a_min'] = [-1.0]*abst_state_size
prnt_pol_params['a_max'] = [1.0]*abst_state_size
prnt_pol_params['init_var'] = 0.1

swrm_pol_params = {}
swrm_pol_params['h_size'] = 100
swrm_pol_params['use_bias'] = False
swrm_pol_params['activ'] = 'leakyrelu_0.1'
swrm_pol_params['a_min'] = [v_min]
swrm_pol_params['a_max'] = [v_max]
swrm_pol_params['n_ders'] = 0
swrm_pol_params['individualized'] = False
swrm_pol_params['parameterized'] = True
swrm_pol_params['max_swarm_size'] = max_swarm_size
swrm_pol_params['init_var'] = 0.1

full_trn_params = {}
full_trn_params['max_swarm_size'] = max_swarm_size
full_trn_params['gamma_abs'] = gamma_abs
full_trn_params['gamma_swm'] = gamma_swm
full_trn_params['gamma_des'] = gamma_des
full_trn_params['gamma_bnd'] = gamma_bnd
full_trn_params['Q_p'] = Q_p
full_trn_params['R_i'] = R_i
full_trn_params['n_steps'] = n_steps
full_trn_params['L'] = p_max - p_min
#full_trn_params['gamma_l1'] = 0.01
full_trn_params['gamma_l2'] = 0.01
full_trn_params['trainer'] = full_trainer
full_trn_params['prnt_state_gen'] = prnt_state_gen
full_trn_params['prnt_param_gen'] = prnt_param_gen
full_trn_params['swrm_state_gen'] = swrm_state_gen
full_trn_params['swrm_param_gen'] = swrm_param_gen
full_trn_params['regen_data'] = regen_data
full_trn_params['verbosity'] = verbosity
full_trn_params['autosave'] = True
full_trn_params['abs_enc_autosave_file'] = lambda epoch:'autosave_epoch%d_abs_enc.pkl'%(epoch,)
full_trn_params['abs_dyn_autosave_file'] = lambda epoch:'autosave_epoch%d_abs_dyn.pkl'%(epoch,)
full_trn_params['prt_pol_autosave_file'] = lambda epoch:'autosave_epoch%d_prt_pol.pkl'%(epoch,)
full_trn_params['swm_pol_autosave_file'] = lambda epoch:'autosave_epoch%d_swm_pol.pkl'%(epoch,)
full_trn_params['patience'] = patience
full_trn_params['use_best'] = use_best
full_trn_params['lr_min'] = lr_min
full_trn_params['lr_max'] = lr_max
full_trn_params['beta'] = beta
full_trn_params['lr_count_patience'] = lr_count_patience
