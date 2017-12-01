#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: olhwdb_client.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-30 17:38:43
# Last Modified: 2017-12-01 15:40:55
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from PIL import Image
import numpy as np
from struct import pack, unpack

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 1

channel = implementations.insecure_channel("127.0.0.1", 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "olhwdb"
request.model_spec.signature_name = "top5"


def parse_image(fn):
    image = tf.image.decode_image(tf.read_file(fn), channels=1)
    image.set_shape([None, None, 1])
    image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image


def preprocess_image(image):
    image = tf.image.per_image_standardization(image)
    return image


filename = "/home/aib/tmp/tui.png"
dataset = tf.data.Dataset.from_tensor_slices([tf.constant(filename)])
dataset = dataset.map(parse_image)
dataset = dataset.map(preprocess_image)
iterator = dataset.make_one_shot_iterator()
images = iterator.get_next()
images = tf.reshape(images, [-1])
with tf.Session() as sess:
    img = sess.run(images)
    proto = tf.make_tensor_proto(values=img)
    request.inputs["image"].CopyFrom(proto)
    result = stub.Predict(request, 10.0)
    classes = result.outputs["classes"].int_val
    scores = result.outputs["scores"].float_val
    assert len(classes) == len(scores)
    for i, cls in enumerate(classes):
        print(pack('<H', cls).decode('gb2312'), scores[i])
