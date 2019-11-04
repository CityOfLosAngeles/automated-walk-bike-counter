# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image
import argparse
import cv2

from .base import BaseController
from ...utils import file_utils as utils
from ...core.tracking.object_tracker import ObjectTracker
from ...core.configuration import config
from ..widgets.aoi import AOIDialog
from ..video import Video,OutputVideo

class MainController(BaseController):

    def __init__(self, view=None, model=None):
        super(MainController, self).__init__(view, model)
        self.view_video_player_widget = None
        self.mask = []
        self.valid_selected_objects = []
        self.video = None
        self.output_video = OutputVideo(self.video)
        self.listener_object = None
        # Should be corrected....
        self.video_settings_pane = None
        self.video_frame = None

    def open_file(self):
        self.video = Video(filedialog.askopenfilename())
        self.video_settings_pane.initialize_resolution_combo()
        self.video_frame.set_progressbar_maximum()

    def update_video_canvas(self,filename,listener_object):

        parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
        parser.add_argument("--anchor_path", type=str, default="https://automated-walk-bike-counter.s3-us-west-1.amazonaws.com/yolo/yolo_anchors.txt",
                            help="The path of the anchor txt file.")
        parser.add_argument("--new_size", nargs='*', type=int, default=[400, 400],
                            help="Resize the input image with `new_size`, size format: [width, height]")
        parser.add_argument("--class_name_path", type=str, default="https://automated-walk-bike-counter.s3-us-west-1.amazonaws.com/yolo/coco.names",
                            help="The path of the class names.")
        parser.add_argument("--restore_path", type=str, default="s3://automated-walk-bike-counter/yolo/yolov3.ckpt",
                           help="The path of the weights to restore.")
        parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=True,
                            help="Whether to save the video detection results.")

        self.listener_object = listener_object
        object_classes,color_table = self.get_selected_objects_list()
        tracker = ObjectTracker(self.mask)
        tracker.video_filename = self.video.filename
        tracker.object_classes = object_classes
        tracker.color_table = color_table
        tracker.video = self.video
        tracker.output_video = self.output_video
        # tracker.frame_listener = self.handle_post_processed_frame
        print(listener_object.handle_post_processed_frame)
        tracker.frame_listener = listener_object.handle_post_processed_frame
        tracker.trackObjects(parser)

    # def handle_post_processed_frame(self,frame):
    #     self.view.handle_post_processed_frame(frame)

    def add_new_aoi(self):
        if self.video:
            aoi_dialog = AOIDialog(self.view.parent, self.video.filename, self)
        else:
            messagebox.showwarning("Warning", "Please select a file!")


    def show_mask(self):
        cv2.imshow("image",self.mask)

    def refresh_aoi_status(self):
        self.view.update_setting_aoi_status()

    def get_selected_objects_list(self):

        objects = []
        colors = {}
        for item in self.valid_selected_objects:
            if item[-1]==1:
                objects.append(item[0].lower())
                color_bgr = item[1][0]
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                colors[item[0].lower()]=color_rgb

        if "cyclist" in objects:
            objects.append("bicycle")
            colors["bicycle"] = (255,255,255)

        print(str(objects))
        print(str(colors))
        return objects,colors

    def cancel_tracking_process(self):
        if self.listener_object:
            self.listener_object.stop_threads()









