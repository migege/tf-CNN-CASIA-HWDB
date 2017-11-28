#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: olhwdb.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-24 16:14:52
# Last Modified: 2017-11-28 18:51:19
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import struct
import os, sys, argparse

import sample_data

flags = tf.app.flags
flags.DEFINE_string("action", None, "[train|evaluate|predict|export]")
flags.DEFINE_string("input", None, "input image path, required when --action=predict")
flags.DEFINE_string("export_dir", None, "model export dir,required when --action=export")
FLAGS = flags.FLAGS

trn_bin = "/home/aib/datasets/OLHWDB1.1trn_pot.bin"
tst_bin = "/home/aib/datasets/OLHWDB1.1tst_pot.bin"
trn_charset = "/home/aib/datasets/OLHWDB1.1trn_pot.bin.charset"

all_tagcodes, all_chars = sample_data.get_all_tagcodes_from_charset_file(trn_charset)
num_classes = len(all_tagcodes)

LABEL_BYTES = 2
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 1
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
RECORD_BYTES = LABEL_BYTES + IMAGE_BYTES


def preprocess_image(image):
    image = tf.image.per_image_standardization(image)
    return image


def parse_record(raw_record):
    record_vector = tf.decode_raw(raw_record, out_type=tf.uint16, little_endian=False, name='decode_raw_16')
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.cast(tf.equal(label, all_tagcodes), tf.int32)
    record_vector = tf.decode_raw(raw_record, out_type=tf.uint8, name='decode_raw_8')
    image = tf.cast(tf.transpose(tf.reshape(record_vector[LABEL_BYTES:RECORD_BYTES], [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH]), [1, 2, 0]), tf.float32)
    return image, label


def input_fn(is_training, batch_size, num_epochs=1):
    if is_training:
        filenames = [trn_bin]
    else:
        filenames = [tst_bin]
    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes=RECORD_BYTES)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.map(parse_record)
    dataset = dataset.map(lambda image, label: (preprocess_image(image), label))
    dataset = dataset.map(lambda image, label: {"image": image}, label)
    dataset = dataset.prefetch(2 * batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


def parse_image(fn):
    image = tf.image.decode_image(tf.read_file(fn), channels=1)
    image.set_shape([None, None, 1])
    image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image


def predict_input_fn(filename):
    dataset = tf.data.Dataset.from_tensor_slices([tf.constant(filename)])
    dataset = dataset.map(parse_image)
    dataset = dataset.map(preprocess_image)
    dataset = dataset.map(lambda image: {"image": image})
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    return images


def CNN(inputs, mode):
    inputs = inputs["image"]
    inputs = tf.reshape(inputs, [-1, 64, 64, 1])
    conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout, units=num_classes)
    return logits


def model_fn(features, labels, mode, params):
    logits = CNN(features, mode)
    predictions = {
        'classes': tf.argmax(logits, 1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        vals, indices = tf.nn.top_k(tf.nn.softmax(logits), k=5)
        classes = tf.gather(all_chars, indices)
        export_outputs = {"top5": tf.estimator.export.ClassificationOutput(classes=classes, scores=vals)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs,
        )

    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    tf.identity(loss, name='train_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, 1), predictions['classes'])
    metrics = {'accuracy': accuracy}
    tf.identity(accuracy[1], name='train_accuracy')

    tensors_to_log = {
        'train_loss': 'train_loss',
        'train_accuracy': 'train_accuracy',
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        training_hooks=[logging_hook],
    )


def main(_):
    if not FLAGS.action or FLAGS.action not in ["train", "evaluate", "predict", "export"]:
        print("--action must be specified.")
        sys.exit(1)
    if FLAGS.action == 'predict' and (not FLAGS.input or not os.path.isfile(FLAGS.input)):
        print("--input must be specified.")
        sys.exit(1)
    if FLAGS.action == 'export' and not FLAGS.export_dir:
        print("--export_dir must be specified.")
        sys.exit(1)

    num_epochs = 5
    epochs_per_eval = 1
    batch_size = 500
    batch_size_evaluate = 1000
    keep_prob = 0.5
    learning_rate = 1e-3

    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=1e4)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="/home/aib/models/tf-CNN-CASIA-OLHWDB/",
        config=run_config,
        params={
            'learning_rate': learning_rate,
            'num_classes': num_classes,
            'keep_prob': keep_prob,
        },
    )

    if FLAGS.action == 'train':
        for _ in range(num_epochs // epochs_per_eval):
            classifier.train(input_fn=lambda: input_fn(True, batch_size, num_epochs))
            eval_results = classifier.evaluate(input_fn=lambda: input_fn(False, batch_size_evaluate))
            print(eval_results)
    elif FLAGS.action == 'evaluate':
        eval_results = classifier.evaluate(input_fn=lambda: input_fn(False, batch_size_evaluate))
        print(eval_results)
    elif FLAGS.action == 'predict':
        for predict_results in classifier.predict(input_fn=lambda: predict_input_fn(FLAGS.input)):
            idx = predict_results['classes']
            print(struct.pack('<H', all_tagcodes[idx]).decode('gb2312'), predict_results['probabilities'][idx])
    elif FLAGS.action == 'export':
        feature_spec = {"image": tf.placeholder(tf.float32, [None, None])}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        classifier.export_savedmodel(FLAGS.export_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
