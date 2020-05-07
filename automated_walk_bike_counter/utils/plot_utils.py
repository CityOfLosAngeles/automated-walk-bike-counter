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

from __future__ import division, print_function

import random

import cv2


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    """
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    """
    print("<<<<<<<<" + label + " -- " + str(color) + ">>>>>>>>>>>>>>>>>>>>")
    tl = line_thickness or int(round(0.004 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = 1
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            float(tl) / 6,
            [0, 0, 0],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
