from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
 
# argument parse for image director path
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", required=True, help="image directory")
args = vars(parser.parse_args())

##initiate hog descriptor
#hog = cv2.HOGDescriptor()
##define customized hog descriptor
hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)

##for 64x128
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
##for 48x96 Daimler detector descriptor
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

# loop through the images
for imagePath in paths.list_images(args["images"]):
    ##read into np array
	image = cv2.imread(imagePath)
    ##for use Daimler detector, no need to resize
	#image = imutils.resize(image, width=1080)
	orig = image.copy()
 
	# detect people in the image
	#(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
	#	padding=(8, 8), scale=1.05)
	#(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
	#	padding=(32, 32), scale=1.05)
	#(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
    #           padding=(16, 16), scale=1.02)
	(rects, weights) = hog.detectMultiScale(image, winStride=(16, 16),
		padding=(32, 32), scale=1.02)
    
	# draw bounding boxes in original image
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw bounding boxes 
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
 
	# show number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
    
    # write out to result/ 
	img = Image.fromarray(image) 
	savepath = "./masked_detect/" + "marked_" + str(filename)
	img.save(savepath)
