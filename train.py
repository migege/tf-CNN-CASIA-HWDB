#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: train.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-15 23:51:22
# Last Modified: 2017-11-17 23:04:46
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import xrange
from PIL import Image
from struct import pack, unpack
import numpy as np
import os
import tensorflow as tf

import sample_data
import model

trn_gnt_bin = "/home/aib/datasets/HWDB1.1trn_gnt.bin"
tst_gnt_bin = "/home/aib/datasets/HWDB1.1tst_gnt.bin"
model_path = "/home/aib/models/tf-CNN-CASIA-HWDB/model.ckpt"

char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
tag_in = map(lambda x: unpack('<H', x.encode('gb2312'))[0], char_set)
assert len(char_set) == len(tag_in)

learning_rate = 1e-3
epochs = 50
batch_size = 500
batch_size_test = 5000
step_display = 10
step_save = 100
p_keep_prob = 0.5
normalize_image = True
one_hot = True
# n_classes = 3755
n_classes = len(tag_in)

x = tf.placeholder(tf.float32, [None, 4096])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

pred = model.CNN(x, n_classes, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())

    if True:
        i = 0
        for epoch in xrange(epochs):
            for batch_x, batch_y in sample_data.read_data_sets(trn_gnt_bin, batch_size=batch_size, normalize_image=normalize_image, tag_in=tag_in, one_hot=one_hot):
                _, summary = sess.run([optimizer, merged_summary_op], feed_dict={x: batch_x, y: batch_y, keep_prob: p_keep_prob})
                summary_writer.add_summary(summary, i)
                i += 1
                if i % step_display == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    print("iters:%s\tloss:%s\taccuracy:%s" % (i * batch_size, "{:.6f}".format(loss), "{:.5f}".format(acc)))
                if i % step_save == 0:
                    saver.save(sess, model_path)
        print("training done.")
        saver.save(sess, model_path)
    else:
        saver.restore(sess, model_path)
        print("model restored.")

    for batch_x, batch_y in sample_data.read_data_sets(tst_gnt_bin, batch_size=batch_size, normalize_image=normalize_image, tag_in=tag_in, one_hot=one_hot):
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        print("test accuracy:{:.5f}".format(acc))
