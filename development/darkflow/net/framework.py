# Modified by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from darkflow.net import framework as fr
from darkflow.net import yolo
from darkflow.net import yolov2
from darkflow.net import vanilla
from .yolov2 import predict
from os.path import basename

class framework(fr.framework):
    pass

class YOLO(fr.YOLO):
    pass

class YOLOv2(fr.YOLOv2):
    postprocess = predict.postprocess
    

"""
framework factory
"""

types = {
    '[detection]': YOLO,
    '[region]': YOLOv2
}

def create_framework(meta, FLAGS):
    net_type = meta['type']
    this = types.get(net_type, framework)
    return this(meta, FLAGS)