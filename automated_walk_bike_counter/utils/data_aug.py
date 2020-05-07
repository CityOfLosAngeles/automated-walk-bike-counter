# Copyright (c) Data Science Research Lab at California State University Los
# Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

# part of this is take from Gluon's repo:
# https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/presets/yolo.py

from __future__ import division, print_function

import random

import cv2
import numpy as np


def resize_with_bbox(img, bbox, new_width, new_height, interp=0):
    """
    Resize the image and correct the bbox accordingly.
    """
    ori_height, ori_width = img.shape[:2]
    img = cv2.resize(img, (new_width, new_height), interpolation=interp)

    # xmin, xmax
    bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
    # ymin, ymax
    bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

    return img, bbox
