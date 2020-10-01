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

from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread

import cv2


class Stream(ABC):
    def __init__(self, stream_source):
        self.stream_source_path = stream_source
        self.camera = None
        self.initialize_camera()
        self.width = int(self.camera.get(3))
        self.height = int(self.camera.get(4))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.area_of_not_interest_mask = []
        # self.line_of_interest_mask = []
        # self.line_of_interest_points = []
        # self.line_of_interest_mask_resized = []
        self.line_of_interest_info = None

    def initialize_camera(self):
        self.camera = cv2.VideoCapture(self.stream_source_path)

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def more(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError


class VideoFile(Stream):
    def __init__(self, filename):
        super().__init__(filename)
        self.frame_count = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stopped = False

    def read(self):
        (grabbed, frame) = self.camera.read()

        if not grabbed:
            self.stopped = True
            self.camera.release()
            return
        else:
            return frame

    def more(self):
        return not self.stopped

    def stop(self):
        self.stopped = True
        self.camera.release()
        self.initialize_camera()


class VideoStream(Stream):
    def __init__(self, stream_url):
        super().__init__(stream_url)
        self.frame_count = 1000000
        self.stopped = False
        self.queue = Queue()
        self.start()

    def start(self):

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):

        while True:

            if self.stopped:
                return

            (grabbed, frame) = self.camera.read()

            if not grabbed:
                self.stop()
                return

            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Queue.empty():
                    pass

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def more(self):
        return not self.stopped

    def stop(self):
        self.stopped = True


class OutputVideo:
    def __init__(self, stream):
        self.original_stream = stream
        self.resolution = None
        self.has_AOI = False
        self.AOI_output_present = False
        self.opaque = 0
