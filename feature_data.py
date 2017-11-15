#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: input_data.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-15 12:36:43
# Last Modified: 2017-11-15 19:02:48
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from six.moves import xrange

from tensorflow.python.platform import gfile


def _read(bytestream, dt, length):
    return np.frombuffer(bytestream.read(length), dtype=dt)[0]


def _extract_header(f):
    _size_of_header = _read(f, np.dtype('<i4'), 4)
    _length_of_illustration = _size_of_header - 62
    _format_code = _read(f, np.dtype('a8'), 8)
    _illustration = _read(f, np.dtype('a%d' % _length_of_illustration), _length_of_illustration)
    _code_type = _read(f, np.dtype('a20'), 20)
    _code_length = _read(f, np.dtype('<i2'), 2)
    _data_type = _read(f, np.dtype('a20'), 20)
    _sample_number = _read(f, np.dtype('<i4'), 4)
    _dimensionality = _read(f, np.dtype('<i4'), 4)
    return _size_of_header, _length_of_illustration, _format_code, _illustration, _code_type, _code_length, _data_type, _sample_number, _dimensionality


def _extract_records(f, record_number, code_length, dimensionality, data_type):
    data_length_map = {
        'unsigned char': 1,
    }

    for i in xrange(record_number):
        label = _read(f, np.dtype(np.uint16), int(code_length))
        buf = f.read(dimensionality * data_length_map[data_type])
        data = np.frombuffer(buf, dtype=np.uint8)
        print('0x%02X' % label, data.shape)


def read_file(fn):
    with gfile.Open(fn, 'rb') as f:
        _, _, _, _, code_type, code_length, data_type, sample_number, dimensionality = _extract_header(f)
        _extract_records(f, sample_number, code_length, dimensionality, data_type)


def read_directory(_dir):
    pass


read_file("/home/aib/datasets/HWDB1.0trn/001.mpf")
