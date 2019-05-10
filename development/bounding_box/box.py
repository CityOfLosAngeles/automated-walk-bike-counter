# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from . import detected_object

class box:
	def __init__(self,left, right, top, bot, mess, max_index, confidence):
		self.left = left
		self.right = right
		self.top = top
		self.bot = bot
		self.mess = mess
		self.max_index = max_index
		self.confidence = confidence

