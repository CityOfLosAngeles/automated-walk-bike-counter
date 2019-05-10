# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from development.bounding_box.detected_object import DetectedObject

class Biker(DetectedObject):
	def __init__(self,box):
		DetectedObject.__init__(self,box)