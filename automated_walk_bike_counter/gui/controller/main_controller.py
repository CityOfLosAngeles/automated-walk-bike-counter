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

from tkinter import BooleanVar, IntVar, filedialog, messagebox

from ...core.configuration import config
from ...core.tracking.object_tracker import ObjectTracker
from ...model.video.video import OutputVideo, VideoFile
from ..view.open_stream_dialog import OpenStreamDialog
from ..widgets.aoi import AOIDialog, AONIDialog, LOIDialog
from .base import BaseController


class MainController(BaseController):
    def __init__(self, view=None, model=None):
        super(MainController, self).__init__(view, model)
        self.view_video_player_widget = None
        self.mask = []
        self.valid_selected_objects = []
        self.stream = None
        self.output_video = OutputVideo(self.stream)
        self.listener_object = None
        # Should be corrected....
        self.video_settings_pane = None
        self.video_frame = None
        self.input_camera_type = "file"
        self.camera_id = 0
        self.internal_webcam = IntVar()
        self.external_webcam = IntVar()
        self.stop_thread = BooleanVar()
        self.tracker = None

    def open_file(self):
        file_name = filedialog.askopenfilename()
        if file_name:
            self.stream = VideoFile(file_name)
            self.input_camera_type = "file"
            self.camera_id = 0
            # Uncheck the selected cameras in source menu
            self.internal_webcam.set(0)
            self.external_webcam.set(0)
            self.video_settings_pane.initialize_resolution_combo()
            self.video_frame.set_progressbar_maximum()

    def update_video_canvas(self, filename, listener_object):
        self.stop_thread.set(False)
        self.listener_object = listener_object
        object_classes, color_table = self.get_selected_objects_list()
        self.tracker = ObjectTracker(self.mask)
        # if self.video:
        #     self.tracker.video_filename = self.video.filename
        self.tracker.valid_selected_objects = [
            "pedestrian" if item == "person" else item
            for item in object_classes
            if item != "bicycle"
        ]
        self.tracker.object_classes = object_classes
        self.tracker.color_table = color_table
        self.tracker.stream = self.stream
        self.tracker.input_camera_type = self.input_camera_type
        self.tracker.camera_id = self.camera_id
        self.tracker.stop_thread = self.stop_thread
        self.tracker.output_video = OutputVideo(self.stream)
        self.tracker.frame_listener = listener_object.handle_post_processed_frame
        self.tracker.track_objects(config)

    def add_new_aoi(self):
        if self.stream:
            AOIDialog(self.view.parent, self.stream.stream_source_path, self)
        else:
            messagebox.showwarning("Warning", "Please select a file!")

    def add_new_aoni(self):
        if self.video:
            AONIDialog(self.view.parent, self.stream.stream_source_path, self)
        else:
            messagebox.showwarning("Warning", "Please select a file!")

    def add_new_loi(self):
        if self.stream:
            LOIDialog(self.view.parent, self.stream.stream_source_path, self)
        else:
            messagebox.showwarning("Warning", "Please select a file!")

    def refresh_aoi_status(self):
        self.view.update_setting_aoi_status()

    def get_selected_objects_list(self):

        objects = []
        colors = {}
        for item in self.valid_selected_objects:
            if item[-1] == 1:
                objects.append(item[0].lower())
                color_bgr = item[1][0]
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                colors[item[0].lower()] = color_rgb

        if "cyclist" in objects:
            objects.append("bicycle")
            colors["bicycle"] = (255, 255, 255)

        return objects, colors

    def cancel_tracking_process(self):
        self.stop_thread.set(True)
        self.stream.stop()
        if self.listener_object:
            self.listener_object.stop_threads()

    def open_stream(self):
        OpenStreamDialog(self.view.parent, self)
