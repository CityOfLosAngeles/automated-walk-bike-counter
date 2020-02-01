# Copyright (c) Data Science Research Lab at California State University Los
# Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import cv2


class Video:
    def __init__(self, filename):
        self.filename = filename
        camera = cv2.VideoCapture(self.filename)
        self.width = int(camera.get(3))
        self.height = int(camera.get(4))
        self.frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = camera.get(cv2.CAP_PROP_FPS)


class OutputVideo:
    def __init__(self, video):
        self.original_video = video
        self.resolution = None
        self.has_AOI = False
        self.AOI_output_present = False
        self.opaque = 0
