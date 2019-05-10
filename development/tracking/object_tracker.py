# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import numpy as np
from development.bounding_box import detected_object
import tensorflow as tf
import math
import cv2
import re
import os
from time import time as timer
from munkres import Munkres,print_matrix
from development.tracking.counter import Object_Counter

import sys
from ..movingobject import MovingObject
from development.frame import Frame


class ObjectTracker:

	PED_AREA_THRESHOLD = 600
	COST_THREASHOLD = 80
	#MISSING_THREASHOLD = 90
	MISSING_THREASHOLD = 90
	MISSING_THREASHOLD_MAX = 300
	BOUNDRY = 30

	COUNT_THRESHOLD = 10
	COUNT_THRESHOLD_BIKE=5
	COUNT_THRESHOLD_MOTOR = 3
	
	def __init__(self):
		self.lastFrameMovingObjects = []
		self.moving_object_id_number = 0
		self.object_counter = Object_Counter()
		self.current_frame = None
		self.currentFrameNumber = 0   # elapsed frames
		self.video_width = 0
		self.video_height = 0
		self.roiBasePoint = []




	def printDataReportOnFrame(self):
		if(self.current_frame!=None):
			h,w = self.current_frame.postprocessed_frame.shape[:2]
			counter = "Pedestrians: " +  str(self.object_counter.COUNTER_p) + " Cyclists: "+ str(self.object_counter.COUNTER_c) + " frame: " + str(self.currentFrameNumber)
			cv2.putText(self.current_frame.postprocessed_frame,counter,(20,int(h-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)


	def addNewMovingObjectToCounter(self,obj,position_new,postprocessed):
		self.object_counter.addNewMovingObjectForCounting(obj,position_new,postprocessed)

		
	def addNewMovingObject(self,current_detected_object):
		self.moving_object_id_number += 1
		new_mObject = MovingObject(self.moving_object_id_number,current_detected_object.center)
		new_mObject.last_detected_object = current_detected_object
		new_mObject.add_position([current_detected_object.center])
		new_mObject.init_kalman_filter()
		filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
		new_mObject.set_next_mean(filtered_state_means[-1])
		new_mObject.set_next_covariance(filtered_state_covariances[-1])
		new_mObject.counted += 1

		##add to current_tracks
		self.lastFrameMovingObjects.append(new_mObject)


	def createContourForCurrentObjects(self,detected_objects):

		detectedObjectsWithValidContours = []

		for obj in detected_objects:
			(left, right, top, bot, mess, max_indx, confidence) = obj.box
			##04/03/2018 exclude pedestrian inside of car by bbox size and shape

			if(mess=='person' and (((right-left)*(bot-top) < self.PED_AREA_THRESHOLD) or (bot-top) < (right-left) * 1.5)):  
				continue                                                                                                     

			detectedObjectsWithValidContours.append(obj)

		return detectedObjectsWithValidContours
		

	def calculateCostMatrixForMovingObjects(self, currentDetectedObjects):

		def get_costs(pos,cur_detected_objects):
			distances = []
			for obj in cur_detected_objects:
				distances.append(math.floor(math.sqrt((obj.center_x-pos[0])**2+(obj.center_y-pos[1])**2)))
			return distances

		def get_costs_extended(last_frame_detected,cur_detected_objects):
			distances = []
			coefficient = 1
			for obj in cur_detected_objects:
				distances.append(math.floor(math.sqrt((obj.center_x - last_frame_detected.predicted_position[-1][0])**2 +
											( obj.center_y - last_frame_detected.predicted_position[-1][1] )**2))+
											coefficient * math.floor(math.sqrt((obj.getWidth()-last_frame_detected.last_detected_object.getWidth())**2+
											(obj.getHeigth()-last_frame_detected.last_detected_object.getHeigth())**2
											)))
			return distances

		lastFrameMovingObjectsCostMatrix = []
		validMovingObjects = []
		for index, obj in enumerate(self.lastFrameMovingObjects):
			##calculate costs for each tracked movingObjects using their predicted position

			costs = get_costs_extended(obj, currentDetectedObjects)

			## if moving object to all contours distances are too large, then not to consider it at all
			if all(c > self.COST_THREASHOLD for c in costs):
				##update it with KF predicted position
				obj.kalman_update_missing(obj.predicted_position[-1])
				##skip this moving object
				continue

			lastFrameMovingObjectsCostMatrix.append(costs)
			##only valid moving objects are added to available_objecs
			validMovingObjects.append(obj)

		return lastFrameMovingObjectsCostMatrix,validMovingObjects


	def update_skipped_frame(self,thresh):

		self.printDataReportOnFrame()

		self.removeTrackedObjects(thresh)
		print("update skipped frame")


	def trackObjects(self, tfnetObject, file, SaveVideo ):

		print("video : ", file)



		# check if the video is reading from a file or from the webcam
		if file == 'camera':
			file = 0
			vfname = "camera"
		else:
			# get the video name to process
			m = re.match(r"([^\.]*)(\..*)",file)
			vfname = m.group(1)

			assert os.path.isfile(file), \
			'file {} does not exist'.format(file)
		
		camera = cv2.VideoCapture(file)

		if file == 0:
			tfnetObject.say('Press [ESC] to quit demo')
			
		assert camera.isOpened(), \
		'Cannot capture source'

		if file == 0:#camera window
			cv2.namedWindow('', 0)
			_, frame = camera.read()
			height, width, _ = frame.shape
			cv2.resizeWindow('', width, height)
		else:
			_, frame = camera.read()
			##if resize:
			##    resize(frame, w=width_new)
			height, width, _ = frame.shape

		self.video_width = width
		self.video_height = height

		if SaveVideo:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			outfile = vfname + "_result.mp4"
			print(outfile)
			if file == 0:#camera window
				fps = 1 / tfnetObject._get_fps(frame)
				if fps < 1:
					fps = 1
			else:
				fps = round(camera.get(cv2.CAP_PROP_FPS))

			videoWriter = cv2.VideoWriter(
				outfile, fourcc, fps, (width, height))

		start = timer()
		n = 0

		## buffers for demo in batch
		buffer_inp = list()
		buffer_pre = list()

		while camera.isOpened():
			self.currentFrameNumber += 1

			elapsed = self.currentFrameNumber

			_, frame = camera.read()
			if frame is None:
				print ('\nEnd of Video')
				break
			
			preprocessed = tfnetObject.framework.preprocess(frame)

			buffer_inp.append(frame)
			buffer_pre.append(preprocessed)

			# Only process and imshow when queue is full
			if elapsed % tfnetObject.FLAGS.queue == 0:
				feed_dict = {tfnetObject.inp: buffer_pre}
				

				net_out = tfnetObject.sess.run(tfnetObject.out, feed_dict)
				
				print("Frame: ", elapsed)
				for img, single_out in zip(buffer_inp, net_out):

					(postprocessed, boxes) = tfnetObject.framework.postprocess(single_out, img, False)

					self.current_frame = Frame(postprocessed, boxes)			

					nodup_objects = self.current_frame.getNoDuplicateObjects()

					noins_objects = self.current_frame.removeObjectsInsideOtherObjects(nodup_objects)
					
					detected_objects = self.createContourForCurrentObjects(noins_objects)

					##first moving object
					if(len(self.lastFrameMovingObjects) == 0):
						#for cont in contours:
						for obj in noins_objects:
							self.addNewMovingObject(obj)

							
					##from the 2nd frame, calculate cost using predicted position and new contour positions
					else:
						##save all the positions to a matrix and calculate the cost
						##initiate a matrix for calculating assianment by Hungarian algorithm
						##when no contour found in this frame then update kalman filter and skip

						if(len(detected_objects) == 0):
							n = n + 1
							self.update_skipped_frame(self.MISSING_THREASHOLD)

							if SaveVideo:
								videoWriter.write(self.current_frame.postprocessed_frame)
							if file == 0: #camera window
								cv2.imshow('', self.current_frame.postprocessed_frame)

							continue

						# matrix_h is the distance between all moving objects of previous frame and the current frame moving objects' centers
						matrix_h,cur_frame_available_moving_objects = self.calculateCostMatrixForMovingObjects(detected_objects)

						## when matrix is empty, skip this frame
						if(len(matrix_h) < 1):

							n = n + 1
							self.update_skipped_frame(self.MISSING_THREASHOLD)

							if SaveVideo:
								videoWriter.write(self.current_frame.postprocessed_frame)
							if file == 0: #camera window
								cv2.imshow('', self.current_frame.postprocessed_frame)

							continue

						self.predictMovingObjectsNewPosition(matrix_h,cur_frame_available_moving_objects,detected_objects)				

					self.printDataReportOnFrame()	


					if SaveVideo:
						videoWriter.write(self.current_frame.postprocessed_frame)
					if file == 0: #camera window
						cv2.imshow('', self.current_frame.postprocessed_frame)
				# Clear Buffers
				buffer_inp = list()
				buffer_pre = list()

			if elapsed % 5 == 0:
				sys.stdout.write('\r')
				sys.stdout.write('{0:3.3f} FPS'.format(
					elapsed / (timer() - start)))
				sys.stdout.flush()
			if file == 0: #camera window
				choice = cv2.waitKey(1)
				if choice == 27: break

		if SaveVideo:
			videoWriter.release()
		camera.release()
		if file == 0: #camera window
			cv2.destroyAllWindows()

		count = "Pedestrians: " +  str(self.object_counter.COUNTER_p) + " Cyclists: "+ str(self.object_counter.COUNTER_c)
		print(count)


	def removeTrackedObjects(self,thresh):
		#MISSING_THREASHOLD = 90
		
		for index, obj in enumerate(self.lastFrameMovingObjects):
			obj.frames_since_seen += 1

			##if a moving object hasn't been updated for 10 frames then remove it
			if obj.frames_since_seen > thresh:
				del self.lastFrameMovingObjects[index]

			## if the object is out of the scene then remove from current tracking right away
			h,w = self.current_frame.postprocessed_frame.shape[:2]
			if (obj.position[-1][0] < 0 or obj.position[-1][0] > w):
				del self.lastFrameMovingObjects[index]

			elif (obj.position[-1][1] < 0 or obj.position[-1][1] > h):
				del self.lastFrameMovingObjects[index]


	
	def predictMovingObjectsNewPosition(self,costMatrix,availableTrackedMovingObjects,cur_detected_objects):

		munkres = Munkres()

		indexes = munkres.compute(costMatrix)

		total = 0
		for row, column in indexes:
			value = costMatrix[row][column]
			total += value
			
		indexes_np = np.array(indexes)

		contour_index_list = indexes_np[:,1].tolist()

		for index_c,obj in enumerate(cur_detected_objects):

			if index_c in indexes_np[:,1]:

				index_m = contour_index_list.index(index_c)

				index_track = indexes_np[index_m,0]

				if costMatrix[index_track][index_c] > self.COST_THREASHOLD:
					self.addNewMovingObject(obj)
					continue


				obj_m = availableTrackedMovingObjects[index_track]
				##get corresponding contour position, update kalman filter
				position_new = cur_detected_objects[index_c].center
				obj_m.last_detected_object = cur_detected_objects[index_c]
				obj_m.kalman_update(position_new)
				obj_m.counted += 1
				print("counted " + str(obj_m.id) + " " + str(obj_m.counted))

				self.addNewMovingObjectToCounter(obj_m,position_new, self.current_frame.postprocessed_frame)
					
								
			else:
				position_new = cur_detected_objects[index_c]

				self.addNewMovingObject(position_new)

		
			##these are tracks missed either because they disappeared 
			## or because they are temporarily invisable 
			for index,obj in enumerate(availableTrackedMovingObjects):
				if index not in indexes_np[:,0]:
					## not update in this frame, increase frames_since_seen
					obj.frames_since_seen += 1
					##but we update KF with predicted location
					obj.kalman_update_missing(obj.predicted_position[-1])
			
			##remove movingObj not updated for more than threasholds numbers of frames  
			for index, obj in enumerate(self.lastFrameMovingObjects):

				h,w = self.current_frame.postprocessed_frame.shape[:2]
				if obj.frames_since_seen > self.MISSING_THREASHOLD:
					if(obj.position[-1][0] < self.BOUNDRY or obj.position[-1][0] > w-self.BOUNDRY or obj.position[-1][1] < self.BOUNDRY or obj.position[-1][1] > h-self.BOUNDRY):
						print("Delete tracking", obj.position[-1][0], obj.position[-1][1])
						del self.lastFrameMovingObjects[index]
					elif (obj.frames_since_seen > self.MISSING_THREASHOLD_MAX):
						print("Delete tracking over max missing threshold")
						del self.lastFrameMovingObjects[index]

	
	def checkObjectForDeletion(self,obj,index):

		if obj.frames_since_seen > self.MISSING_THREASHOLD:
			if(obj.position[-1][0] < self.BOUNDRY or obj.position[-1][0] > self.video_width-self.BOUNDRY or obj.position[-1][1] < self.BOUNDRY or obj.position[-1][1] > self.video_height-self.BOUNDRY):
				print("Delete tracking", obj.position[-1][0], obj.position[-1][1])
				del self.lastFrameMovingObjects[index]
			elif (obj.frames_since_seen > self.MISSING_THREASHOLD_MAX):
				print("Delete tracking over max missing threshold")
				del self.lastFrameMovingObjects[index]




