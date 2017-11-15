#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: sample_data.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-15 22:53:41
# Last Modified: 2017-11-16 00:30:59
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np


def read_from_gnt_dir(gnt_dir):

    def one_file(f):
        header_size = 10
        while True:
            _sample_size = np.fromfile(f, np.dtype('<u4'), 1)
            if not _sample_size.size:
                break
            sample_size = _sample_size[0]
            tagcode = np.fromfile(f, np.dtype('<u2'), 1)[0]
            width = np.fromfile(f, np.dtype('<u2'), 1)[0]
            height = np.fromfile(f, np.dtype('<u2'), 1)[0]
            if header_size + width * height != sample_size:
                break
            img = np.fromfile(f, np.uint8, width * height).reshape((height, width))
            yield tagcode, img

    for fn in os.listdir(gnt_dir):
        if fn.endswith(".gnt"):
            fn = os.path.join(gnt_dir, fn)
            with open(fn, 'rb') as f:
                for tagcode, img in one_file(f):
                    yield tagcode, img


def extract_first_100_images(gnt_dir):
    i = 0
    for tagcode, img in read_from_gnt_dir(gnt_dir):
        try:
            tag = struct.pack('<H', tagcode).decode('gb2312')
            i += 1
        except:
            continue
        print('0x%04x' % tagcode, tag, img.shape)
        png = Image.fromarray(img)
        png.convert('RGB').save('./png/' + tag + str(i) + '.png')
        if i > 100:
            break


def resize_and_normalize_image(img):
    import scipy.misc

    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    img = scipy.misc.imresize(img, (64 - 4 * 2, 64 - 4 * 2))
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert img.shape == (64, 64)

    img = img.flatten()
    img = (img - 128) / 128
    return img


def read_data_sets(gnt_dir):
    x = []
    y = []
    y_uniq = []
    for tagcode, img in read_from_gnt_dir(gnt_dir):
        x.append(resize_and_normalize_image(img))
        y.append(tagcode)
        if tagcode not in y_uniq:
            y_uniq.append(tagcode)
    print("tagcode len:", len(y_uniq))
    for i, v in enumerate(y):
        new_v = np.zeros(len(y_uniq))
        new_v[y_uniq.index(v)] = 1
        y[i] = new_v
    return x, y
