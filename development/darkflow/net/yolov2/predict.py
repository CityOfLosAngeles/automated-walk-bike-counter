# Modified by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import numpy as np
import math
import cv2
import os
import json

from darkflow.utils.box import BoundBox
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	resultsForJSON = []
	results = []
	for b in boxes:

		boxResults = self.process_box(b, h, w, threshold)

		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue
		if(mess == 'person' or mess == 'bicycle' or mess == 'motorbike'):

			results.append(boxResults)			
			cv2.rectangle(imgcv,
				(left, top), (right, bot),
				colors[max_indx], thick)
			cv2.putText(imgcv, mess, (left, top - 12),
				0, 1e-3 * h, colors[max_indx],thick//3)


	if not save: return imgcv, results

	return imgcv, results
