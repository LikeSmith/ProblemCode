"""
Model core class

A Model is a object that is a trainable mapping from input to output
to train, a model that inherits from this should have the following:
    self.build_network: returns inputs, outputs, weights, losses, and training
                        ops.  Takes set of inputs as optional argument.

for saving and loading, it is also recommended that self.weights be a
dictionary that contains all the weights of the network
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import math
from tqdm import trange
import os
import errno

try:
    import _pickle as pkl
except ImportError:
    import pickle as pkl

class Model(object):
    def __init__(self, params={}, load_file=None, home='', name='base_model', sess=None, debug=False):
        self.home = home
        self.name = name

        # Generate or load network
        self.assign_op = []

        if load_file is not None:
            self.params, values = self.load(load_file, assign=False)
        else:
            self.params = params

        self.inputs, self.output_ops, self.weights, self.loss_ops, self.train_ops = self.build_network()

        if sess is None:
            if debug:
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            else:
                self.sess = tf.Session()
        else:
            self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        if load_file is not None:
            self.assign_weights(values)

    def build_network(self, inputs=None, weights=None):
        raise NotImplementedError

    def get_save_params(self):
        return self.params

    def train(self, train_data, val_data, val_size=None, n_epochs=100, n_batches=None, batch_size=128, patience=10, verbosity=2, use_best=True, shuffle=False, no_regen=False):
        '''
        generic training algorithm with early stopping, can be over written for more specialized
        cases.
        '''

        train_data = batch_generator(train_data, self.inputs, n_batches, batch_size, shuffle, no_regen)
        val_data = batch_generator(val_data, self.inputs, 1, batch_size, shuffle, True)[0]

        n_batches = train_data.n_batches

        # train network
        hist = {}
        for key in self.output_ops.keys():
            shape_arr = self.sess.run(tf.shape(self.output_ops[key][0]))
            shape = ()
            for dim in shape_arr:
                if dim > 0:
                    shape += (dim,)

            hist[key] = np.zeros((n_epochs, val_size)+shape)

        for key in self.loss_ops.keys():
            hist['ave_%s'%key] = np.zeros(n_epochs)
            hist['val_%s'%key] = np.zeros(n_epochs)

        min_weights={}
        min_loss = math.inf
        min_loss_epoch = -1
        min_count = 0

        if verbosity > 0:
            print('Training...')
        if verbosity == 3:
            epochs = trange(n_epochs)
            epochs.set_description('Val. Loss: N/A')
        else:
            epochs = range(n_epochs)

        for epoch in epochs:
            if verbosity in [1, 2]:
                print('Epoch %d/%d'%(epoch+1, n_epochs))

            if verbosity in [2, 3]:
                batches = trange(n_batches)
                batches.set_description('Ave. Loss: N/A')
            else:
                batches = range(n_batches)

            for batch in batches:

                loss, _ = self.sess.run([self.loss_ops, self.train_ops], feed_dict=train_data[batch])

                for key in self.loss_ops.keys():
                    hist['ave_%s'%key][epoch] += loss[key]

                if verbosity in [2, 3]:
                    msg = ''
                    for key in self.loss_ops.keys():
                        msg += '(ave_%s: %f)'%(key, hist['ave_%s'%key][epoch]/(batch+1))

                    batches.set_description(msg)

            val_outputs, val_loss = self.sess.run([self.output_ops, self.loss_ops], feed_dict=val_data)

            total_val_loss = 0

            for key in self.output_ops.keys():
                hist[key][epoch, :] = val_outputs[key]

            for key in self.loss_ops.keys():
                hist['ave_%s'%key][epoch] /= n_batches
                hist['val_%s'%key][epoch] = val_loss[key]
                total_val_loss += val_loss[key]

            if verbosity in [1, 2]:
                msg = ''
                for key in self.loss_ops.keys():
                    msg += '(ave_%s: %f, val_%s: %f)'%(key, hist['ave_%s'%key][epoch], key, hist['val_%s'%key][epoch])

                print('Epoch %d complete! %s'%(epoch+1, msg))

            elif verbosity == 3:
                msg = ''
                for key in self.loss_ops.keys():
                    msg += '(val_%s: %f)'%(key, hist['val_%s'%key][epoch])
                epochs.set_description(msg)

            if total_val_loss < min_loss:
                min_loss = total_val_loss
                min_loss_epoch = epoch
                min_weights = self.save_weights()
                min_count = 0
            else:
                min_count += 1

            if min_count > patience:
                print('Validation loss has stopped decreassing, stopping training early.')

                for key in self.output_ops.keys():
                    hist[key] = hist[key][:epoch+1, :]

                for key in self.loss_ops.keys():
                    hist['ave_%s'%key] = hist['ave_%s'%key][:epoch+1]
                    hist['val_%s'%key] = hist['val_%s'%key][:epoch+1]

                break;

        if use_best:
            self.assign_weights(min_weights)

        hist['min_loss'] = min_loss
        hist['min_loss_epoch'] = min_loss_epoch

        return hist

    def predict(self, input_feed):
        outputs = self.sess.run(self.output_ops, feed_dict=input_feed)
        return outputs

    def __call__(self, inputs):
        is_tf = False
        for val in inputs.values():
            if isinstance(val, tf.Tensor):
                is_tf = True
                break;

        if is_tf:
            return self.build_network(inputs)
        else:
            return self.predict(inputs)

    def save(self, filename):
        weight_vals = self.save_weights()

        if self.home is not '':
            try:
                os.makedirs(self.home, exist_ok=True)
            except TypeError:
                try:
                    os.makedirs(self.home)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise

        pkl.dump({'params':self.get_save_params(), 'weights':weight_vals}, open(self.home+filename, 'wb'))

    def load(self, filename, assign=True):
        load_dat = pkl.load(open(self.home+filename, 'rb'))
        if assign:
            self.assign_weights(load_dat['weights'])
        return load_dat['params'], load_dat['weights']

    def save_weights(self):
        weight_vals = {}
        for key in self.weights.keys():
            weight_vals[key] = self.sess.run(self.weights[key])
        return weight_vals

    def assign_weights(self, values):
        assign_ops = {}
        for key in self.weights.keys():
            assign_ops[key] = self.weights[key].assign(values[key])

        self.sess.run(assign_ops)

    def lock(self):
        weights = self.save_weights()
        const_weights = {}
        for key in weights.keys():
            const_weights[key] = tf.constant(weights[key], name=key)

        self.inputs, self.output_ops, self.weights, self.loss_ops, self.train_ops = self.build_network(inputs=self.inputs, weights=const_weights)


class batch_generator(object):
    def __init__(self, data, inputs, n_batches, batch_size, shuffle=False, no_regen=False):
        n_samp = -1

        for key, value in data.items():
            if n_samp == -1 and not callable(value):
                n_samp = value.shape[0]
            elif not callable(value) and n_samp != value.shape[0]:
                raise DataSizeMissMatch()

            if shuffle and not callable(value):
                perm = np.random.permutation(n_samp)
                data[key] = value[perm]

        if batch_size is None:
            batch_size = n_samp

        # calculate total number of batches
        if n_batches is None:
            n_batches = math.ceil(n_samp/batch_size)

        # break up data into batches and validation set
        bat_feed = []

        for i in range(n_batches):
            bat_feed.append({})

        for key, value in data.items():
            for i in range(n_batches-1):
                if callable(value):
                    bat_feed[i][key] = value
                else:
                    bat_feed[i][key] = value[i*batch_size:(i+1)*batch_size]
            if callable(value):
                bat_feed[-1][key] = value
            else:
                bat_feed[-1][key] = value[(n_batches-1)*batch_size:]

        if no_regen:
            for key, value in data.items():
                if callable(value):
                    for i in range(n_batches):
                        bat_feed[i][key] = value(i, batch_size, n_batches, n_samp)

        self.bat_feed = bat_feed
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_samp = n_samp
        self.data_keys = data.keys()
        self.inputs = inputs

    def __getitem__(self, i):
        bat = {}
        for key, value in self.bat_feed[i].items():
            if callable(value):
                bat[self.inputs[key]] = value(i, self.batch_size, self.n_batches, self.n_samp)
            else:
                bat[self.inputs[key]] = value

        return bat

    def keys(self):
        return self.data_keys

class DataSizeMissMatch(Exception):
    pass
