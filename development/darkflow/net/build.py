# Modified by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import tensorflow as tf
import time
import json
from .framework import create_framework
from darkflow.dark.darknet import Darknet

from darkflow.net.build import TFNet
from development import main

class NewTFNet(TFNet):

    count = main.count

    def __init__(self, FLAGS, darknet = None):
    	self.ntrain = 0

    	if isinstance(FLAGS, dict):
    		from darkflow.defaults import argHandler
    		newFLAGS = argHandler()
    		newFLAGS.setDefaults()
    		newFLAGS.update(FLAGS)
    		FLAGS = newFLAGS

    	self.FLAGS = FLAGS
    	if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
    		self.say('\nLoading from .pb and .meta')
    		self.graph = tf.Graph()
    		device_name = FLAGS.gpuName \
    			if FLAGS.gpu > 0.0 else None
    		with tf.device(device_name):
    			with self.graph.as_default() as g:
    				self.build_from_pb()
    		return

    	if darknet is None:	
    		darknet = Darknet(FLAGS)
    		self.ntrain = len(darknet.layers)

    	self.darknet = darknet
    	args = [darknet.meta, FLAGS]
    	self.num_layer = len(darknet.layers)
    	self.framework = create_framework(*args)
		
    	self.meta = darknet.meta

    	self.say('\nBuilding net ...')
    	start = time.time()
    	self.graph = tf.Graph()
    	device_name = FLAGS.gpuName \
    		if FLAGS.gpu > 0.0 else None
    	with tf.device(device_name):
    		with self.graph.as_default() as g:
    			self.build_forward()
    			self.setup_meta_ops()
    	self.say('Finished in {}s\n'.format(
    		time.time() - start))
	
    def build_from_pb(self):
    	with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
    		graph_def = tf.GraphDef()
    		graph_def.ParseFromString(f.read())
		
    	tf.import_graph_def(
    		graph_def,
    		name=""
    	)
    	with open(self.FLAGS.metaLoad, 'r') as fp:
    		self.meta = json.load(fp)
    	self.framework = create_framework(self.meta, self.FLAGS)

    	# Placeholders
    	self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
    	self.feed = dict() # other placeholders
    	self.out = tf.get_default_graph().get_tensor_by_name('output:0')
		
    	self.setup_meta_ops()