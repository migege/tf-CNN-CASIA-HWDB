#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: sample_data.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-15 22:53:41
# Last Modified: 2017-11-24 09:38:31
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import codecs
from sklearn.utils import shuffle
from struct import pack, unpack


def read_from_pot_dir(pot_dir):

    def one_file(f):
        header_size = 8
        while True:
            _sample_size = np.fromfile(f, np.dtype('<u2'), 1)
            if not _sample_size:
                break
            sample_size = _sample_size[0]
            tagcode = np.fromfile(f, np.dtype('<u4'), 1)[0]
            stroke_num = np.fromfile(f, np.dtype('<u2'), 1)[0]
            strokes = []
            one_stroke = []
            while True:
                x = np.fromfile(f, np.dtype('<i2'), 1)[0]
                y = np.fromfile(f, np.dtype('<i2'), 1)[0]
                if x == -1 and y == 0:
                    strokes.append(one_stroke)
                    one_stroke = []
                    continue
                if x == -1 and y == -1:
                    yield tagcode, strokes
                    break
                one_stroke.append((x, y))

    for fn in os.listdir(pot_dir):
        if fn.endswith('.pot'):
            fn = os.path.join(pot_dir, fn)
            with open(fn, 'rb') as f:
                for tagcode, strokes in one_file(f):
                    yield tagcode, strokes


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


def resize_image(img):
    import scipy.misc

    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.pad(img, pad_dims, mode='constant', constant_values=255)
    img = scipy.misc.imresize(img, (64 - 4 * 2, 64 - 4 * 2))
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert img.shape == (64, 64)

    img = img.flatten()
    return img


def normalize_img(img):
    img = (img - 128) / 128
    return img


def get_all_tagcodes(gnt_bin):
    with open(gnt_bin, 'rb') as f:
        tagcode_all = []
        while True:
            buf = np.fromfile(f, np.uint8, 4098)
            if not buf.size:
                break
            tagcode = np.frombuffer(buf, np.dtype('<u2'), 1)[0]
            if tagcode not in tagcode_all:
                tagcode_all.append(tagcode)
        return tagcode_all


def get_all_tagcodes_from_charset_file(fn):
    with codecs.open(fn, 'r', encoding='utf8') as f:
        tagcode_all = []
        chars = f.read().strip()
        for ch in chars:
            tagcode_all.append(unpack('<H', ch.encode('gb2312'))[0])
        return tagcode_all


def read_data_sets(gnt_bin, batch_size=50, normalize_image=True, tag_in=[], one_hot=True):
    with open(gnt_bin, 'rb') as f:
        x = []
        y = []
        while True:
            buf = np.fromfile(f, np.uint8, 4098)
            if not buf.size:
                break

            tagcode = np.frombuffer(buf, np.dtype('<u2'), 1)[0]
            if tagcode not in tag_in:
                continue

            if one_hot:
                label = np.zeros(len(tag_in))
                label[tag_in.index(tagcode)] = 1
            else:
                label = tagcode

            image = np.frombuffer(buf, np.uint8, 4096)
            if normalize_image:
                image = normalize_img(image)
            x.append(image)
            y.append(label)
            assert len(x) == len(y)
            if len(x) == batch_size:
                x, y = shuffle(x, y, random_state=0)
                _x = np.array(x[:])
                _y = np.array(y[:])
                x = []
                y = []
                yield _x, _y


if __name__ == '__main__':
    print(get_all_tagcodes_from_charset_file("/home/aib/datasets/OLHWDB1.1trn_pot.bin.charset"))
