# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import inspect
import numpy as np
from pykalman import KalmanFilter

class MovingObject(object):

    def __init__(self,id,position):
        self.id = id
        self.position = np.array([position])
        self.predicted_position = np.array([position])
        self.frames_since_seen = 0
        self.frame_since_start = 0
        self.counted = 0
        self.counted_biker = 0
        self.counted_moter = 0
        self.counted_car = 0
        self.counted_bus = 0
        self.counted_truck = 0
        self.counted_ped = 0 
        self.next_mean = []
        self.next_covariance = []
        self.detection = 0
        self.pedestrian_id = -1
        self.last_detected_object = None

        
    def count_frames(self):
        self.frame_since_start = self.frame_since_start + 1
        return self.frame_since_start
    
    def add_position(self, position_new):
        self.position = np.append(self.position, position_new, axis=0)
        self.frames_since_seen = 0

        
    def add_predicted_position(self, next_position):
        self.predicted_position = np.append(self.predicted_position, next_position, axis=0)
        
    def init_kalman_filter(self):
        transition_matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
        observation_matrix=[[1,0,0,0],[0,1,0,0]]
        initstate = [0,0,0,0]
        initial_state_covariance = [[2,0,0,0],[0,2,0,0],[0,0,0.5,0],[0,0,0,0.5]]
        ## value of Q 
        transition_covariance = [[5,0,0,0],[0,5,0,0],[0,0,2,0],[0,0,0,0.5]]
        ##Value of R
        #observation_covariance = [[5,0,0,0],[0,5,0,0],[0,0,2,0],[0,0,2,0]]
        observation_covariance =[[5,0],[0,5]]
        self.kf = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initstate,
                  initial_state_covariance = initial_state_covariance,
                  transition_covariance = transition_covariance,
                  observation_covariance= observation_covariance
                  #em_vars=['transition_covariance', 'initial_state_covariance']
                )
    
    def kalman_update(self, new_position):
        next_mean = self.next_mean
        next_covariance = self.next_covariance
        next_mean, next_covariance = self.kf.filter_update(
            filtered_state_mean = next_mean, 
            filtered_state_covariance = next_covariance,
            observation = new_position)
        self.set_next_mean(next_mean)
        self.set_next_covariance(next_covariance)
        self.add_predicted_position([self.next_mean[:2]])
        self.frames_since_seen = 0
    
    def kalman_update_missing(self, new_position):
        previous_mean = self.next_mean
        next_mean = self.next_mean
        next_covariance = self.next_covariance
        next_mean, next_covariance = self.kf.filter_update(
            filtered_state_mean = next_mean, 
            filtered_state_covariance = next_covariance,
            observation = new_position)
        ##since we don't want to change speed, adjust the mean value for delta x and delta y
        adjusted_next_mean = np.concatenate((next_mean[:2], previous_mean[2:]),axis=0)
        self.set_next_mean(adjusted_next_mean)
        self.set_next_covariance(next_covariance)
        self.add_predicted_position([self.next_mean[:2]])

        #self.update_last_detected_object_values(self.next_mean[:2])
        self.frames_since_seen += 1
    
    def set_next_mean(self,mean):
        self.next_mean = mean
    def set_next_covariance(self,covariance):
        self.next_covariance = covariance
    
    def set_frames_since_seen(self, num):
        self.frames_since_seen = num
        
    def detection_increase(self):
        self.detection += 1

    def update_last_detected_object_values(self,newCenterPosition):
        center_x,center_y = newCenterPosition
        w = self.last_detected_object.left - self.last_detected_object.right
        h = self.last_detected_object.top - self.last_detected_object.bot

        new_left = int(center_x - w/2)
        new_top = int(center_y - h/2)
        new_right = int(center_x + w/2)
        new_bot = int(center_y + h/2)

        self.last_detected_object.left = new_left
        self.last_detected_object.right = new_right
        self.last_detected_object.top = new_top
        self.last_detected_object.bot = new_bot

        
