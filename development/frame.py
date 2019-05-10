# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from development.bounding_box.pedestrian import Pedestrian
from development.bounding_box.biker import Biker
from development.bounding_box.motorbiker import MotorBiker
import numpy as np
import cv2

class Frame:
	##duplicated classification threshold > 0.30 then means 2 classification for same object
	DUPLICATE_THREASHOLD = 0.30
	BIK_AREA_THRESHOLD = 700

	def __init__(self,postprocessed,boxes):
		self.pedestrians = []
		self.bikers = []
		self.motorbikers = []
		self.postprocessed_frame = postprocessed
		self.boxes = boxes
		self.createDetectedObject()

	def addPedestrian(self,pedestrian):
		self.pedestrians.append(pedestrian)

	def addBiker(self,biker):
		self.bikers.append(biker)

	def addMotorbiker(self,motorbiker):
		self.motorbikers.append(motorbiker)

	def createDetectedObject(self):
		for box in self.boxes:
			(left, right, top, bot, mess, max_indx, confidence) = box

			if(mess == 'person'):
				self.pedestrians.append(Pedestrian(box))
			elif(mess == 'bicycle'):
				self.bikers.append(Biker(box))
			elif(mess == 'motorbike'):
				self.motorbikers.append(MotorBiker(box))


	def findDuplicateObjects(self):
	    
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
		
		ped_boxes_dup_dict = {}
		##compare bike/motorbike and pedestrian, find duplicates
		for bik in self.bikers:
			for ped in self.pedestrians:
				boxes_2compare = np.array([[ped.left,ped.top,ped.right,ped.bot],[bik.left,bik.top,bik.right,bik.bot]])
				o_rate = overlap_area(boxes_2compare)
				print("overlap: ", o_rate)
				if(o_rate > self.DUPLICATE_THREASHOLD):
					ped_boxes_dup_dict[ped] =1
					print("exclude for duplicates:",ped.left, ped.right, ped.top, ped.bot )

		for mot in self.motorbikers:
			for ped in self.pedestrians:
				boxes_2compare = np.array([[ped.left,ped.top,ped.right,ped.bot],[mot.left,mot.top,mot.right,mot.bot]])
				o_rate = overlap_area(boxes_2compare)
				print("overlap: ", o_rate)
				if(o_rate > self.DUPLICATE_THREASHOLD):
					ped_boxes_dup_dict[ped] =1
					print("exclude for duplicates:",ped.left, ped.right, ped.top, ped.bot )

		return ped_boxes_dup_dict

	def getNoDuplicateObjects(self):
	    
		noDuplicateObjects = []
		ped_boxes_dup_dict = self.findDuplicateObjects()

		##add pedestrian only into nodup_boxes
		for ped in self.pedestrians:                 
			if ped in ped_boxes_dup_dict:
				continue
			else:
				noDuplicateObjects.append(ped)
                
		##add bike into nodup_boxes
		for bik in self.bikers:
			##add filter to exclude bicycle bbox too small, could be a false detection of a handbag
			if((bik.right-bik.left)*(bik.bot-bik.top) < self.BIK_AREA_THRESHOLD and (bik.bot-bik.top) < (bik.right-bik.left)*1.5 ):
				continue
			else:
				noDuplicateObjects.append(bik)
                
		##add motorbikers 7/27, since we need to do better job for excluding motorbikers
		for mot in self.motorbikers:
			noDuplicateObjects.append(mot)

		return noDuplicateObjects

	def removeObjectsInsideOtherObjects(self,listOfObjects):
		insider_dict = {}
		noInside_Objects = []

		margin = 10
		##remove any box that inside of other boxes
		for nbox in listOfObjects:
			for mbox in listOfObjects:
				if(nbox.box == mbox.box):
					continue
				if(mbox.left+margin >= nbox.left and mbox.right-margin <= nbox.right and mbox.top+margin>=nbox.top and mbox.bot-margin<=nbox.bot):
					insider_dict[mbox] = 1 ## ---- item caused different counter
					print(mbox.left, mbox.right, mbox.top, mbox.bot," inside of ", nbox.left, nbox.right, nbox.top, nbox.bot)
                            
		for obj in listOfObjects:
			if obj in insider_dict:
				continue
			else:
				noInside_Objects.append(obj)

		return noInside_Objects