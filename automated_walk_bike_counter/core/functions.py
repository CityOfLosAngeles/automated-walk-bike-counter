# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import logging
import math
from . import MovingObject

#function overlap calculate how much 2 rectangles overlap
def overlap_area(boxes):
    if(len(boxes) == 0):
        return 0
    
    xx1 = max(boxes[0,0], boxes[1,0])
    yy1 = max(boxes[0,1], boxes[1,1])
    xx2 = min(boxes[0,2], boxes[1,2]) 
    yy2 = min(boxes[0,3], boxes[1,3])
    
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    
    ##box1 area
    area1 = (boxes[0,2]-boxes[0,0]) * (boxes[0,3]-boxes[0,1])
    ##box2 area
    area2 = (boxes[1,2]-boxes[1,0]) * (boxes[1,3]-boxes[1,1])
    
    area = min(area1,area2)    
    overlap = (w * h) / area
    
    return overlap

##calculate distance between 2 points, pos: [0,0], points: [[1,1],[2,2],[3,3]]
def get_costs(pos,points):
    distances = [math.floor(math.sqrt((x2-pos[0])**2+(y2-pos[1])**2)) for (x2,y2) in points]
    return distances

    ###for logging
def mylogger(name, logfile):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger = logging.getLogger(name)
        hdlr = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

    return logger

def update_skipped_frame(frame,fname,tracks,thresh):

    h,w = frame.shape[:2]
    #counter_p = "Pedestrians: " +  str(COUNTER_p) + " frame " + fname
    #counter_c = "Cyclists: " +  str(COUNTER_c) + " frame " + fname
    #cv2.putText(frame,counter_p,(int(w/4),int(h-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
    #cv2.putText(frame,counter_c,(int(w/4),int(h-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)

    ##update tracking objects
    for obj in tracks:
        obj.frames_since_seen += 1
    new_tracks = removeTrackedObjects(tracks,frame,thresh)
    print("update skipped frame")

    return new_tracks

def removeTrackedObjects(tracking_arr,frame,thresh):
    #MISSING_THREASHOLD = 90
    for index, obj in enumerate(tracking_arr):
    ##if a moving object hasn't been updated for 10 frames then remove it
        if obj.frames_since_seen > thresh:
            del tracking_arr[index]
        ## if the object is out of the scene then remove from current tracking right away
        h,w = frame.shape[:2]
        if (obj.position[-1][0] < 0 or obj.position[-1][0] > w):
            del tracking_arr[index]
        elif (obj.position[-1][1] < 0 or obj.position[-1][1] > h):
            del tracking_arr[index]

    return tracking_arr
    
def track_new_object(position, tracks, counter):
    new_mObject = MovingObject(counter,position)
    new_mObject.add_position([position])
    new_mObject.init_kalman_filter()
    filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
    new_mObject.set_next_mean(filtered_state_means[-1])
    new_mObject.set_next_covariance(filtered_state_covariances[-1])

    ##add to current_tracks
    tracks.append(new_mObject)
        

