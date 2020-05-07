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

# for better understanding about yolov3 architecture, refer to this website (in Chinese)
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf

from ..utils.layer_utils import conv2d, darknet53_body, upsample_layer, yolo_block

slim = tf.contrib.slim


class yolov3(object):
    def __init__(
        self,
        class_num,
        anchors,
        use_label_smooth=False,
        use_focal_loss=False,
        batch_norm_decay=0.999,
        weight_decay=5e-4,
    ):

        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            "decay": self.batch_norm_decay,
            "epsilon": 1e-05,
            "scale": True,
            "is_training": is_training,
            "fused": None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope(
                [slim.conv2d],
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params,
                biases_initializer=None,
                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                weights_regularizer=slim.l2_regularizer(self.weight_decay),
            ):
                with tf.variable_scope("darknet53_body"):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope("yolov3_head"):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(
                        net,
                        3 * (5 + self.class_num),
                        1,
                        stride=1,
                        normalizer_fn=None,
                        activation_fn=None,
                        biases_initializer=tf.zeros_initializer(),
                    )
                    feature_map_1 = tf.identity(feature_map_1, name="feature_map_1")

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(
                        net,
                        3 * (5 + self.class_num),
                        1,
                        stride=1,
                        normalizer_fn=None,
                        activation_fn=None,
                        biases_initializer=tf.zeros_initializer(),
                    )
                    feature_map_2 = tf.identity(feature_map_2, name="feature_map_2")

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(
                        feature_map_3,
                        3 * (5 + self.class_num),
                        1,
                        stride=1,
                        normalizer_fn=None,
                        activation_fn=None,
                        biases_initializer=tf.zeros_initializer(),
                    )
                    feature_map_3 = tf.identity(feature_map_3, name="feature_map_3")

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        """
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3]
            returned from `forward` function
        anchors: shape: [3, 2]
        """
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map)[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [
            (anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors
        ]

        feature_map = tf.reshape(
            feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num]
        )

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self.class_num], axis=-1
        )
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(
            tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32
        )

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        """
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [
            (feature_map_1, self.anchors[6:9]),
            (feature_map_2, self.anchors[3:6]),
            (feature_map_3, self.anchors[0:3]),
        ]
        reorg_results = [
            self.reorg_layer(feature_map, anchors)
            for (feature_map, anchors) in feature_map_anchors
        ]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(
                conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1]
            )
            prob_logits = tf.reshape(
                prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num]
            )
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
