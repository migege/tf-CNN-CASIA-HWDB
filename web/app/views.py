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

from app import app

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 1

channel = implementations.insecure_channel("127.0.0.1", 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


def preprocess_image(img):
    s = img.size
    num_pixels = np.prod(s)
    image = [float(ord(v)) for v in img.tobytes()]
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
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

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
