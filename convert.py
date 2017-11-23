#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: convert.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-16 10:34:55
# Last Modified: 2017-11-23 16:56:16
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import xrange
from struct import pack
from collections import defaultdict
from PIL import Image, ImageDraw
import os
import numpy as np
import sample_data

trn_gnt_dir = "/home/aib/datasets/HWDB1.1trn_gnt/"
tst_gnt_dir = "/home/aib/datasets/HWDB1.1tst_gnt/"

trn_pot_dir = "/home/aib/datasets/OLHWDB1.1trn_pot/"
tst_pot_dir = "/home/aib/datasets/OLHWDB1.1tst_pot/"


def convert_gnt(gnt_dir, fn_dst):
    with open(fn_dst, 'wb') as f:
        for tagcode, img in sample_data.read_from_gnt_dir(gnt_dir):
            tagcode.tofile(f)
            norm_img = sample_data.resize_image(img)
            norm_img.tofile(f)


def extract_pot(pot_dir, png_dir):
    all_tagcode = defaultdict(int)
    for tagcode, strokes in sample_data.read_from_pot_dir(pot_dir):
        all_tagcode[tagcode] += 1
        pngf = os.path.join(png_dir, "%05d_%s.png" % (tagcode, all_tagcode[tagcode]))
        im = Image.new("L", (10240, 10240), 255)
        draw = ImageDraw.Draw(im)
        mins = []
        maxs = []
        for stroke in strokes:
            mins.append(np.min(stroke, 0))
            maxs.append(np.max(stroke, 0))
            draw.line(stroke, fill=0, width=4)
        del draw
        _min = np.min(mins, 0) - 2
        _max = np.max(maxs, 0) + 2
        box = (_min[0], _min[1], _max[0], _max[1])
        im = im.crop(box)
        im.save(pngf)

    chars = [pack('>H', v).decode('gb2312') for v in all_tagcode.keys()]
    with open(os.path.join(png_dir, "charset"), "w") as f:
        for char in chars:
            f.write(char.encode('utf8'))
        f.write("\n")


# convert_gnt(trn_gnt_dir, "/home/aib/datasets/HWDB1.1trn_gnt.bin")
# convert_gnt(tst_gnt_dir, "/home/aib/datasets/HWDB1.1tst_gnt.bin")

extract_pot(trn_pot_dir, "/home/aib/datasets/OLHWDB1.1trn_png/")
extract_pot(tst_pot_dir, "/home/aib/datasets/OLHWDB1.1tst_png/")
