#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: train.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-15 23:51:22
# Last Modified: 2017-11-24 18:50:09
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import xrange
from PIL import Image
from struct import pack, unpack
import numpy as np
import os, sys
import argparse
import tensorflow as tf

import sample_data
import model

FLAGS = None

trn_bin = "/home/aib/datasets/HWDB1.1trn_gnt.bin"
tst_bin = "/home/aib/datasets/HWDB1.1tst_gnt.bin"
model_path = "/home/aib/models/tf-CNN-CASIA-HWDB/model.ckpt"

trn_bin = "/home/aib/datasets/OLHWDB1.1trn_pot.bin"
tst_bin = "/home/aib/datasets/OLHWDB1.1tst_pot.bin"
model_path = "/home/aib/models/tf-CNN-CASIA-OLHWDB/model.ckpt"
trn_charset = "/home/aib/datasets/OLHWDB1.1trn_pot.bin.charset"

learning_rate = 1e-3
epochs = 40
batch_size = 200
batch_size_test = 1000
step_display = 10
step_save = 500
p_keep_prob = 0.5
normalize_image = True
one_hot = True


def main(_):
    FN_model = None

    if FLAGS.charset == 0:
        char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
        tag_in = map(lambda x: unpack('<H', x.encode('gb2312'))[0], char_set)
        assert len(char_set) == len(tag_in)
        n_classes = len(char_set)
        FN_model = model.CNN
    elif FLAGS.charset == 1:
        # tag_in = sample_data.get_all_tagcodes(trn_gnt_bin)
        tag_in = sample_data.get_all_tagcodes_from_charset_file(trn_charset)
        assert len(tag_in) == 3755
        n_classes = len(tag_in)
        FN_model = model.cnn_for_medium_charset

    x = tf.placeholder(tf.float32, [None, 4096])
    y = tf.placeholder(tf.int32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    pred = FN_model(x, n_classes, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    cr5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred, tf.argmax(y, 1), 5), tf.float32))
    cr10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred, tf.argmax(y, 1), 10), tf.float32))

    top5 = tf.nn.top_k(tf.nn.softmax(pred), k=5)

    tf.summary.scalar("loss", cost)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())

        if FLAGS.action == 'train':
            i = 0
            for epoch in xrange(epochs):
                for batch_x, batch_y in sample_data.read_data_sets(trn_bin, batch_size=batch_size, normalize_image=normalize_image, tag_in=tag_in, one_hot=one_hot):
                    _, summary = sess.run([optimizer, merged_summary_op], feed_dict={x: batch_x, y: batch_y, keep_prob: p_keep_prob})
                    summary_writer.add_summary(summary, i)
                    i += 1
                    if i % step_display == 0:
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                        print("batch:%s\tloss:%s\taccuracy:%s" % (i, "{:.6f}".format(loss), "{:.5f}".format(acc)))
                    if i % step_save == 0:
                        saver.save(sess, model_path)
            print("training done.")
            saver.save(sess, model_path)
        else:
            saver.restore(sess, model_path)
            print("model restored.")

        if FLAGS.action == 'test':
            i = 0
            sum_cr1 = 0.
            sum_cr5 = 0.
            sum_cr10 = 0.
            for batch_x, batch_y in sample_data.read_data_sets(tst_bin, batch_size=batch_size_test, normalize_image=normalize_image, tag_in=tag_in, one_hot=one_hot):
                loss, acc, _cr5, _cr10 = sess.run([cost, accuracy, cr5, cr10], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                print("Loss:{:.6f}\tCR(1):{:.5f}\tCR(5):{:.5f}\tCR(10):{:.5f}".format(loss, acc, _cr5, _cr10))
                sum_cr1 += acc
                sum_cr5 += _cr5
                sum_cr10 += _cr10
                i += 1
            print("============================================================")
            print("CR(1):{:.5f}\tCR(5):{:.5f}\tCR(10):{:.5f}".format(sum_cr1 / i, sum_cr5 / i, sum_cr10 / i))
        elif FLAGS.action == 'inference':
            imgs = FLAGS.img.split(';')
            if not imgs:
                raise Exception('--img is invalid')
            for imgf in imgs:
                if not os.path.isfile(imgf):
                    print("file:%s invalid" % imgf)
                    continue
                with Image.open(imgf).convert('L') as img_obj:
                    shape = (img_obj.size[1], img_obj.size[0])
                    img = np.reshape(bytearray(img_obj.tobytes()), shape)
                    img = sample_data.resize_image(img)

                    if normalize_image:
                        img = sample_data.normalize_img(img)
                    vals, indices = sess.run(top5, feed_dict={x: [img], keep_prob: 1.})
                    print("============================================================")
                    for i, index in enumerate(indices[0]):
                        print("%s : %s" % (pack('<H', tag_in[index]).decode('gb2312'), vals[0][i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='[train|test|inference]')
    parser.add_argument('charset', type=int, help='0:only mostly used 140 characters; 1:3755 characters in GB2312')
    parser.add_argument('--img', type=str, help='the image path, required when action=inference')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
