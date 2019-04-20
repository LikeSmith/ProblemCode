"""
Serializer
"""

import tensorflow as tf

activ_items = {}
activ_items['relu'] = tf.nn.relu
activ_items['leakyrelu'] = tf.nn.leaky_relu
activ_items['relu6'] = tf.nn.relu6
activ_items['softmax'] = tf.nn.softmax
activ_items['tanh'] = tf.tanh
activ_items['linear'] = lambda x: x
activ_items['sigmoid'] = tf.sigmoid

class leaky_relu(object):
    def __init__(self, item):
        self.alpha = float(item.split('_')[1])

    def __call__(self, val):
        return tf.nn.leaky_relu(val, alpha=self.alpha)

activ_par_items = {}
activ_par_items['leakyrelu'] = leaky_relu

def activ_deserialize(item, custom_items={}, custom_par_items={}):
    items = {**activ_items, **custom_items}
    par_items = {**activ_par_items, **custom_par_items}

    if isinstance(item, str):
        if item in items.keys():
            return items[item]

        if item.split('_')[0] in par_items.keys():
            return par_items[item.split('_')[0]](item)

    return item
