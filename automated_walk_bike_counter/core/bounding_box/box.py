# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

class box:
	def __init__(self,left, right, top, bot, mess, max_index, confidence):
		self.left = left
		self.right = right
		self.top = top
		self.bot = bot
		self.mess = mess
		self.max_index = max_index
		self.confidence = confidence

