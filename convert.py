#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: convert.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-16 10:34:55
# Last Modified: 2017-11-16 11:24:33
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import xrange
from struct import pack
import numpy as np
import sample_data

trn_gnt_dir = "/home/aib/datasets/HWDB1.1trn_gnt/"
tst_gnt_dir = "/home/aib/datasets/HWDB1.1tst_gnt/"


def convert(gnt_dir, fn_dst):
    with open(fn_dst, 'wb') as f:
        for tagcode, img in sample_data.read_from_gnt_dir(gnt_dir):
            tagcode.tofile(f)
            norm_img = sample_data.resize_image(img)
            norm_img.tofile(f)
            break


convert(trn_gnt_dir, "/home/aib/datasets/HWDB1.1trn_gnt.bin")
convert(tst_gnt_dir, "/home/aib/datasets/HWDB1.1tst_gnt.bin")
