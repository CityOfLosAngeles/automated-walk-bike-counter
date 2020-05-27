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

import math


# function overlap calculate how much 2 rectangles overlap
def overlap_area(boxes):
    if len(boxes) == 0:
        return 0

    xx1 = max(boxes[0, 0], boxes[1, 0])
    yy1 = max(boxes[0, 1], boxes[1, 1])
    xx2 = min(boxes[0, 2], boxes[1, 2])
    yy2 = min(boxes[0, 3], boxes[1, 3])

    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)

    # box1 area
    area1 = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
    # box2 area
    area2 = (boxes[1, 2] - boxes[1, 0]) * (boxes[1, 3] - boxes[1, 1])

    area = min(area1, area2)
    overlap = (w * h) / area

    return overlap


# calculate distance between 2 points, pos: [0,0], points: [[1,1],[2,2],[3,3]]
def get_costs(pos, points):
    distances = [
        math.floor(math.sqrt((x2 - pos[0]) ** 2 + (y2 - pos[1]) ** 2))
        for (x2, y2) in points
    ]
    return distances


def update_skipped_frame(frame, fname, tracks, thresh):

    h, w = frame.shape[:2]

    # update tracking objects
    for obj in tracks:
        obj.frames_since_seen += 1
    new_tracks = remove_tracked_objects(tracks, frame, thresh)
    print("update skipped frame")

    return new_tracks


def remove_tracked_objects(tracking_arr, frame, thresh):
    for index, obj in enumerate(tracking_arr):
        # if a moving object hasn't been updated for 10 frames then remove it
        if obj.frames_since_seen > thresh:
            del tracking_arr[index]
        # if the object is out of the scene then remove from current tracking right away
        h, w = frame.shape[:2]
        if obj.position[-1][0] < 0 or obj.position[-1][0] > w:
            del tracking_arr[index]
        elif obj.position[-1][1] < 0 or obj.position[-1][1] > h:
            del tracking_arr[index]

    return tracking_arr
