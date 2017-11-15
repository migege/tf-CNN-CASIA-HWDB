#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: train.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-15 23:51:22
# Last Modified: 2017-11-16 00:31:23
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import xrange

import struct
from PIL import Image

import sample_data

trn_gnt_dir = "/home/aib/datasets/HWDB1.1trn_gnt/"
tst_gnt_dir = "/home/aib/datasets/HWDB1.1tst_gnt/"

trn_x, trn_y = sample_data.read_data_sets(trn_gnt_dir)
print(len(trn_x), len(trn_y))
