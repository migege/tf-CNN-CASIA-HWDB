#!/usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations

from flask import render_template, json, jsonify, request
from PIL import Image, ImageFont, ImageDraw
from struct import pack, unpack
import base64

from app import app

__global_times = 0
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 1

channel = implementations.insecure_channel("127.0.0.1", 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


def parse_image(fn):
    image = tf.image.decode_image(tf.read_file(fn), channels=1)
    image.set_shape([None, None, 1])
    image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image


def preprocess_image(image):
    image = tf.image.per_image_standardization(image)
    return image


def create_image(char):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fonts = ImageFont.truetype("./app/static/fonts/msyh.ttc", 36, encoding='utf-8')
    dr.text((15, 10), char, font=fonts, fill="#000000")
    del dr
    return im.tobytes()


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title='Home')


@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.form.get('data'))
    imagedata = data["test_image"]
    imagedata = imagedata[22:]
    img = base64.b64decode(imagedata)
    with open(__test_image_file, 'wb') as f:
        f.write(img)

    global __global_times
    if (__global_times == 0):
        global __sess
        __sess = tf.Session()
        __global_times = 1

    dataset = tf.data.Dataset.from_tensor_slices([tf.constant(__test_image_file)])
    dataset = dataset.map(parse_image)
    dataset = dataset.map(preprocess_image)
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    images = tf.reshape(images, [-1])
    img = __sess.run(images)
    proto = tf.make_tensor_proto(values=img)
    request.inputs["image"].CopyFrom(proto)
    result = stub.Predict(request, 10.0)
    classes = result.outputs["classes"].int_val
    scores = result.outputs["scores"].float_val
    assert len(classes) == len(scores)
    info = {}
    for i, cls in enumerate(classes):
        if i > 2:
            break
        char = pack('<H', cls).decode('gb2312')
        prob = scores[i]
        img = base64.b64encode(create_image(char)).decode()
        info['pred%s_image' % (i + 1)] = "data:image/jpg;base64," + img
        info['pred%s_accuracy' % (i + 1)] = str('{:.2%}'.format(prob)
    return jsonify(info)
