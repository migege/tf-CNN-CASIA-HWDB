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
import numpy as np
import base64
import cStringIO
import scipy.misc

from app import app

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 1

channel = implementations.insecure_channel("127.0.0.1", 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


def per_image_standardization(image, size):
    num_pixels = np.prod(size)
    image_mean = np.mean(image)
    variance = np.mean(np.square(image)) - np.square(image_mean)
    variance = np.maximum(variance, 0)
    stddev = np.sqrt(variance)
    min_stddev = np.reciprocal(np.sqrt(num_pixels))
    pixel_value_scale = np.maximum(stddev, min_stddev)
    pixel_value_offset = image_mean
    image = np.subtract(image, pixel_value_offset)
    image = np.divide(image, pixel_value_scale)
    return image


def preprocess_image(img):  # PIL.Image
    s = img.size
    image = [ord(v) for v in img.tobytes()]
    image = np.reshape(image, (s[1], s[0]))
    h_list = []
    w_list = []
    for h, row in enumerate(image):
        for w, pix in enumerate(row):
            if pix != 255:
                h_list.append(h)
                w_list.append(w)
    min_h = max(np.min(h, 0) - 2, 0)
    max_h = min(np.max(h, 0) + 2, s[1])
    min_w = max(np.min(w, 0) - 2, 0)
    max_w = min(np.max(w, 0) + 2, s[0])
    box = (min_w, min_h, max_w, max_h)
    img = img.crop(box)

    s = img.size
    pad_size = abs(s[0] - s[1]) // 2
    image = np.reshape([float(ord(v)) for v in img.tobytes()], (s[1], s[0]))
    if image.shape[0] < image.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    image = np.pad(image, pad_dims, mode='constant', constant_value=255.0)
    image = scipy.misc.imresize(image, (IMAGE_HEIGHT - 4 * 2, IMAGE_WIDTH - 4 * 2))
    image = np.pad(image, ((4, 4), (4, 4)), mode='constant', constant_value=255.0)
    assert image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    image = image.flatten()

    image = per_image_standardization(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return image


def create_image(char):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fonts = ImageFont.truetype("./app/static/fonts/msyh.ttc", 36, encoding='utf-8')
    dr.text((15, 10), char, font=fonts, fill="#000000")
    del dr
    buff = cStringIO.StringIO()
    im.save(buff, format="JPEG")
    return buff.getvalue()


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

    image = Image.open(cStringIO.StringIO(img)).convert('L')
    # if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
    # image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

    img = preprocess_image(image)
    proto = tf.make_tensor_proto(values=img)

    req = predict_pb2.PredictRequest()
    req.model_spec.name = "olhwdb"
    req.model_spec.signature_name = "top5"
    req.inputs["image"].CopyFrom(proto)
    result = stub.Predict(req, 10.0)
    classes = result.outputs["classes"].int_val
    scores = result.outputs["scores"].float_val
    assert len(classes) == len(scores)
    info = {}
    for i, cls in enumerate(classes):
        if i > 2:
            break
        char = pack('<H', cls).decode('gb2312')
        prob = scores[i]
        img = base64.b64encode(create_image(char))
        info['pred%s_image' % (i + 1)] = "data:image/jpg;base64," + img
        info['pred%s_accuracy' % (i + 1)] = str('{:.2%}'.format(prob))
        # print(char, prob)
    return jsonify(info)
