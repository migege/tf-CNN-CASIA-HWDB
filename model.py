#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: model.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-16 11:58:28
# Last Modified: 2017-11-17 15:11:03
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)


def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 90, beta=0.75, name=name)


def DNN(x, n_classes, keep_prob):
    # W_h1 = weights_variable([4096, 2048])
    # W_h2 = weights_variable([2048, 256])
    W_h1 = weights_variable([4096, 256])
    W_o = weights_variable([256, n_classes])

    x = tf.nn.dropout(x, 0.8)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W_h1)), keep_prob)
    # h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, W_h2)), keep_prob)
    # y = tf.matmul(h2, W_o)
    y = tf.matmul(h1, W_o)
    return y


def CNN(x, n_classes, keep_prob):
    x = tf.reshape(x, [-1, 64, 64, 1])

    W_c1 = weights_variable([3, 3, 1, 32])
    b_c1 = biases_variable([32])
    c1 = conv2d('c1', x, W_c1, b_c1)
    p1 = maxpool2d('p1', c1)
    p1 = tf.nn.dropout(p1, keep_prob)

    W_c2 = weights_variable([3, 3, 32, 64])
    b_c2 = biases_variable([64])
    c2 = conv2d('c2', p1, W_c2, b_c2)
    p2 = maxpool2d('p2', c2)
    p2 = tf.nn.dropout(p2, keep_prob)

    W_fc1 = weights_variable([16 * 16 * 64, 1024])
    b_fc1 = biases_variable([1024])
    flat_p2 = tf.reshape(p2, [-1, 16 * 16 * 64])
    fc1 = tf.nn.relu(tf.matmul(flat_p2, W_fc1) + b_fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    W_fc2 = weights_variable([1024, n_classes])
    b_fc2 = biases_variable([n_classes])
    y = tf.matmul(fc1, W_fc2) + b_fc2
    return y
