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

import tensorflow as tf


def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3,
            for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS
            filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype="int32")

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(
        boxes, [-1, 4]
    )  # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(
            boxes=filter_boxes,
            scores=filter_score,
            max_output_size=max_boxes,
            iou_threshold=nms_thresh,
            name="nms_indices",
        )
        label_list.append(
            tf.ones_like(tf.gather(filter_score, nms_indices), "int32") * i
        )
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label
