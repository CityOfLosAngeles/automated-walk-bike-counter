# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

class DetectedObject:

	def __init__(self,box):

		self.box = box
		(left, right, top, bot, mess, max_index, confidence) = box
		self.left = left
		self.right = right
		self.top = top
		self.bot = bot
		self.mess = mess
		self.max_indx = max_index
		self.confidence = confidence
		self.center_x = 0
		self.center_y = 0
		self.center = self.getObjectCenteralPointArray()	

	def getObjectCenteralPointArray(self):
		cx = self.left + int((self.right-self.left)/2)
		self.center_x = cx
		cy = self.top + int((self.bot-self.top)/2)
		self.center_y = cy
		return [cx,cy]

	def getWidth(self):
		return self.right-self.left

	def getHeigth(self):
		return self.bot-self.top


	

	

