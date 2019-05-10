
from time import time as timer
import tensorflow as tf
import numpy as np
import sys
import cv2
import os

from development.darkflow.cli import cliHandler

args = ['./flow', '--model', 'cfg/yolo.cfg', '--load', 'bin/yolo1.weights', '--demo', 'cam/68position_001.mp4','--saveVideo','--gpu','0.5']
#args = ['./flow', '--model', 'cfg/yolo.cfg', '--load', 'bin/yolo1.weights', '--demo', 'camera','--gpu','0.5']                                     # Real time detection
cliHandler(args)
