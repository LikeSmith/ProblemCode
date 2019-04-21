"""
Shooting_AbsTrack.py

Shooting method for tracking abstract states

params:
max_swarm_size
gamma_abs
gamma_swm
gamma_des
gamma_bnd
Q_p
R_i
n_steps
L
gamma_l1_abs
gamma_l2_abs
gamma_l1_swm
gamma_l2_swm
gamma_l1_prt
gamma_l2_prt
trainer
trainer_abs
trainer_swm
trainer_prt
prnt_state_gen
prnt_param_gen
swrm_state_gen
swrm_param_gen
regen_data
verbosity
autosave
abs_enc_autosave_file
abs_dyn_autosave_file
prt_pol_autosave_file
swm_pol_autosave_file
patience
use_best
"""

import tensorflow as tf
import numpy as np
import math
import _pickle as pkl
from tqdm import trange

from .Model import Model

class Shooting_AbsTrack(Model):
    def __init__(self, abs_enc, abs_dyn, prt_pol, swm_pol, prt_dyn, swm_dyn, params={}, load_file=None, home='', name='shooting', debug=False, sess=None):
        self.abs_enc = abs_enc
        self.abs_dyn = abs_dyn
        self.prt_pol = prt_pol
        self.swm_pol = swm_pol
        self.prt_dyn = prt_dyn
        self.swm_dyn = swm_dyn

        self.prnt_state_size = prt_dyn.state_size[0]
        self.prnt_param_size = prt_dyn.param_size[0]
        self.swrm_state_size = swm_dyn.state_size[1]
        self.swrm_actin_size = swm_dyn.actin_size[1]
        self.swrm_param_size = swm_dyn.param_size[1]
        self.abst_state_size = abs_enc.abs_size

        super(Shooting_AbsTrack, self).__init__(params=params, load_file=load_file, home=home, name=name, sess=sess, debug=debug)

        self.abs_enc.sess = self.sess
        self.abs_dyn.sess = self.sess
        self.prt_pol.sess = self.sess
        self.swm_pol.sess = self.sess
        self.prt_dyn.sess = self.sess
        self.swm_dyn.sess = self.sess

        self.sess.run(tf.global_variables_initializer())

    def build_network(self, inputs=None, weights=None):
        with tf.variable_scope(self.name):
            if inputs is None:
                if self.params['verbosity'] > 2:
                    print('building inputs...')
                inputs = {}
                inputs['s_p_0'] = tf.placeholder('float', (None, self.prnt_state_size), 's_p_0')
                inputs['s_s_0'] = tf.placeholder('float', (None, self.params['max_swarm_size'], self.swrm_state_size), 's_s_0')
                inputs['p_p'] = tf.placeholder('float', (None, self.prnt_param_size), 'p_p')
                inputs['p_s'] = tf.placeholder('float', (None, self.params['max_swarm_size'], self.swrm_param_size), 'p_s')
                self.training = tf.placeholder('bool', name='training')

            if weights is None:
                try:
                    weights = self.weights
                except AttributeError:
                    weights = {}
                    for key in self.abs_enc.weights.keys():
                        weights['abs_enc_%s'%key] = self.abs_enc.weights[key]
                    for key in self.abs_dyn.weights.keys():
                        weights['abs_dyn_%s'%key] = self.abs_dyn.weights[key]
                    for key in self.swm_pol.weights.keys():
                        weights['swm_pol_%s'%key] = self.swm_pol.weights[key]
                    for key in self.prt_pol.weights.keys():
                        weights['prt_pol_%s'%key] = self.prt_pol.weights[key]

                    self.gamma_abs = tf.constant(self.params['gamma_abs'], dtype='float')
                    self.gamma_swm = tf.constant(self.params['gamma_swm'], dtype='float')
                    self.gamma_des = tf.constant(self.params['gamma_des'], dtype='float')
                    self.gamma_bnd = tf.constant(self.params['gamma_bnd'], dtype='float')
                    self.Q_p = tf.constant(self.params['Q_p'], dtype='float')
                    self.R_i = tf.constant(self.params['R_i'], dtype='float')

            if self.params['verbosity'] > 2:
                print('setting up for forward pass')

            prnt_states = [inputs['s_p_0']]
            prnt_actins = []
            swrm_states = [inputs['s_s_0']]
            swrm_actins = []
            abst_states = [self.abs_enc.build_abs(inputs['s_s_0'], inputs['p_s'], self.training)]

            l_abs = []
            l_swm = []
            l_prt = []
            l_bnd = []

            dyn_inputs = {}
            dyn_inputs['prnt_params'] = inputs['p_p']
            dyn_inputs['training'] = self.training

            with tf.variable_scope('forward') as scope:
                if self.params['verbosity'] > 2:
                    print('Performing forward pass...')
                    iters = trange(self.params['n_steps'])
                else:
                    iters = range(self.params['n_steps'])
                for i in iters:
                    s_p_cur = prnt_states[i]
                    s_s_cur = swrm_states[i]
                    s_a_cur = abst_states[i]

                    dyn_inputs['prnt_states'] = s_p_cur
                    dyn_inputs['abst_states'] = s_a_cur

                    _, dyn_out, _, _, _ = self.abs_dyn.build_network(dyn_inputs, scope=scope)
                    s_p_prd = dyn_out['prnt_states']

                    a_p_cur = self.prt_pol.build_policy(s_p_cur, inputs['p_p'], i, training=self.training)
                    a_s_cur = self.swm_pol.build_policy(s_s_cur, s_a_cur, a_p_cur, inputs['p_s'], i, training=self.training)

                    s_p_nxt = self.prt_dyn.build_dynamics(s_p_cur, tf.concat((s_s_cur, inputs['p_s']), axis=2), inputs['p_p'], i)
                    s_s_nxt = self.swm_dyn.build_dynamics(s_s_cur, a_s_cur, inputs['p_s'], i)
                    s_a_nxt = self.abs_enc.build_abs(s_s_nxt, inputs['p_s'], self.training)

                    l_abs.append(self.gamma_abs*tf.square(tf.norm(s_p_prd-s_p_nxt, axis=1)))
                    l_swm.append(self.gamma_swm*tf.square(tf.norm(s_a_cur-a_p_cur, axis=1) + tf.einsum('ijk,kl,ijl->i', a_s_cur, self.R_i, a_s_cur)))
                    l_prt.append(tf.einsum('ij,jk,ik->i', s_p_cur, self.Q_p, s_p_cur) + self.gamma_des*tf.square(tf.norm(a_p_cur, axis=1)))
                    l_bnd.append(self.gamma_bnd*tf.reduce_sum(tf.square(tf.maximum(0.0, tf.abs(s_s_nxt[:, :, 0])-self.params['L']/2.0)), axis=1))

                    prnt_states.append(s_p_nxt)
                    prnt_actins.append(a_p_cur)
                    swrm_states.append(s_s_nxt)
                    swrm_actins.append(a_s_cur)
                    abst_states.append(s_a_nxt)

            output_ops = {}
            output_ops['prnt_states'] = tf.stack(prnt_states, axis=1)
            output_ops['prnt_actins'] = tf.stack(prnt_actins, axis=1)
            output_ops['swrm_states'] = tf.stack(swrm_states, axis=1)
            output_ops['swrm_actins'] = tf.stack(swrm_actins, axis=1)
            output_ops['abst_states'] = tf.stack(abst_states, axis=1)
            output_ops['l_abs'] = tf.stack(l_abs, axis=1)
            output_ops['l_swm'] = tf.stack(l_swm, axis=1)
            output_ops['l_prt'] = tf.stack(l_prt, axis=1)
            output_ops['l_bnd'] = tf.stack(l_bnd, axis=1)

            with tf.variable_scope('loss'):
                loss_ops = {}
                loss_ops['L_abs'] = tf.reduce_mean(tf.reduce_sum(output_ops['l_abs'], axis=1))
                loss_ops['L_swm'] = tf.reduce_mean(tf.reduce_sum(output_ops['l_swm'], axis=1))
                loss_ops['L_prt'] = tf.reduce_mean(tf.reduce_sum(output_ops['l_prt'], axis=1))
                loss_ops['L_bnd'] = tf.reduce_mean(tf.reduce_sum(output_ops['l_bnd'], axis=1))

                loss_all = loss_ops['L_abs']+loss_ops['L_swm']+loss_ops['L_prt']+loss_ops['L_bnd']

                if 'gamma_l1' in self.params.keys():
                    gamma_l1 = tf.constant(self.params['gamma_l1'], dtype='float')
                    for key in weights.keys():
                        loss_all += gamma_l1*tf.reduce_sum(tf.abs(weights[key]))
                if 'gamma_l2' in self.params.keys():
                    gamma_l2 = tf.constant(self.params['gamma_l2'], dtype='float')
                    for key in weights.keys():
                        loss_all += gamma_l2*tf.nn.l2_loss(weights[key])

            train_ops = {}
            with tf.variable_scope('train'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    train_ops['train_pol'] = self.params['trainer'].minimize(loss_all)

        self.loss_all = loss_all

        return inputs, output_ops, weights, loss_ops, train_ops

    def train(self, n_epochs, n_batches, batch_size, val_size):
        hist = {}
        var_list = tf.global_variables()

        for key in self.loss_ops.keys():
            hist['ave_%s'%key] = np.zeros(n_epochs)
            hist['val_%s'%key] = np.zeros(n_epochs)

        min_weights={}
        min_loss = math.inf
        min_loss_epoch = -1
        min_count = 0

        s_p_0_val = self.params['prnt_state_gen'](val_size)
        s_s_0_val = self.params['swrm_state_gen'](val_size, self.params['max_swarm_size'])
        p_p_val = self.params['prnt_param_gen'](val_size)
        p_s_val = self.params['swrm_param_gen'](val_size, self.params['max_swarm_size'])
        feed_dict_val={self.inputs['s_p_0']:s_p_0_val, self.inputs['s_s_0']:s_s_0_val, self.inputs['p_p']:p_p_val, self.inputs['p_s']:p_s_val, self.training:False}

        if not self.params['regen_data']:
            s_p_0_list = []
            s_s_0_list = []
            p_p_list = []
            p_s_list = []
            if self.params['verbosity'] > 0:
                print('Generating Training data...')
            if self.params['verbosity'] > 1:
                batches = trange(n_batches)
            else:
                batches = range(n_batches)

            for i in batches:
                s_p_0_list.append(self.params['prnt_state_gen'](batch_size))
                s_s_0_list.append(self.params['swrm_state_gen'](batch_size, self.params['max_swarm_size']))
                p_p_list.append(self.params['prnt_param_gen'](batch_size))
                p_s_list.append(self.params['swrm_param_gen'](batch_size, self.params['max_swarm_size']))

        if self.params['verbosity'] > 0:
            print('Training...')

        for epoch in range(n_epochs):
            print('Epoch %d/%d'%(epoch+1, n_epochs))
            if self.params['verbosity'] > 1:
                batches = trange(n_batches)
                batches.set_description('Ave. Loss: N/A')
            else:
                batches = range(n_batches)

            for batch in batches:
                if self.params['regen_data']:
                    s_p_0 = self.params['prnt_state_gen'](batch_size)
                    s_s_0 = self.params['swrm_state_gen'](batch_size, self.params['max_swarm_size'])
                    p_p = self.params['prnt_param_gen'](batch_size)
                    p_s = self.params['swrm_param_gen'](batch_size, self.params['max_swarm_size'])
                else:
                    s_p_0 = s_p_0_list[batch]
                    s_s_0 = s_s_0_list[batch]
                    p_p = p_p_list[batch]
                    p_s = p_s_list[batch]

                feed_dict={self.inputs['s_p_0']:s_p_0, self.inputs['s_s_0']:s_s_0, self.inputs['p_p']:p_p, self.inputs['p_s']:p_s, self.training:True}

                old_vars = self.sess.run(var_list)

                loss, _ = self.sess.run([self.loss_ops, self.train_ops], feed_dict=feed_dict)

                new_vars = self.sess.run(var_list)
                has_nans = False
                for var in new_vars:
                    if np.isnan(var).any():
                        print('NAAAAAAAAANS')
                        has_nans = True
                        break
                for key in loss.keys():
                    if np.isnan(loss[key]):
                        print('NAAAAAAAAANS')
                        has_nans = True
                        break
                        
                if has_nans:
                    import pdb; pdb.set_trace()
                    continue

                for key in self.loss_ops.keys():
                    hist['ave_%s'%key][epoch] += loss[key]

                if self.params['verbosity'] > 1:
                    msg = ''
                    for key in self.loss_ops.keys():
                        msg += '(ave_%s: %f)'%(key, hist['ave_%s'%key][epoch]/(batch+1))
                    if 'use_lr_search' in self.params.keys() and self.params['use_lr_search']:
                        for key in self.lr_tf.keys():
                            msg += '(%s: %f)'%(key, lr[key])

                    batches.set_description(msg)

            val_outputs, val_loss = self.sess.run([self.output_ops, self.loss_ops], feed_dict=feed_dict_val)

            total_val_loss = 0

            for key in self.loss_ops.keys():
                hist['ave_%s'%key][epoch] /= n_batches
                hist['val_%s'%key][epoch] = val_loss[key]
                total_val_loss += val_loss[key]

            msg = ''
            for key in self.loss_ops.keys():
                msg += '(ave_%s: %f, val_%s: %f)'%(key, hist['ave_%s'%key][epoch], key, hist['val_%s'%key][epoch])

            print('Epoch %d complete! %s'%(epoch+1, msg))

            if total_val_loss < min_loss:
                min_loss = total_val_loss
                min_loss_epoch = epoch
                min_weights = self.save_weights()
                min_count = 0
                if 'autosave' in self.params.keys() and self.params['autosave']:
                    self.abs_enc.save(self.params['abs_enc_autosave_file'](epoch))
                    self.abs_dyn.save(self.params['abs_dyn_autosave_file'](epoch))
                    self.prt_pol.save(self.params['prt_pol_autosave_file'](epoch))
                    self.swm_pol.save(self.params['swm_pol_autosave_file'](epoch))
                    pkl.dump([val_outputs, val_loss, s_p_0_val, s_s_0_val, p_p_val, p_s_val], open(self.home+'val_results_epoch%d.pkl'%epoch, 'wb'))
            else:
                min_count += 1

            if 'patience' in self.params.keys() and min_count > self.params['patience']:
                print('Validation loss has stopped decreassing, stopping training early.')

                for key in self.loss_ops.keys():
                    hist['ave_%s'%key] = hist['ave_%s'%key][:epoch+1]
                    hist['val_%s'%key] = hist['val_%s'%key][:epoch+1]

                break;

        if self.params['use_best']:
            self.assign_weights(min_weights)

        hist['min_loss'] = min_loss
        hist['min_loss_epoch'] = min_loss_epoch

        return hist
