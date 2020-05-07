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

# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import tensorflow as tf

from automated_walk_bike_counter.core.model import YoloV3
from automated_walk_bike_counter.utils.misc_utils import load_weights, parse_anchors

num_class = 80
img_size = 416
weight_path = "./data/yolo_weights/yolov3.weights"
save_path = "./data/yolo_weights/yolov3.ckpt"
anchors = parse_anchors("./data/yolo_anchors.txt")

model = YoloV3(80, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope("yolov3"):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope="yolov3"))

    load_ops = load_weights(tf.global_variables(scope="yolov3"), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print("TensorFlow model checkpoint has been saved to {}".format(save_path))
