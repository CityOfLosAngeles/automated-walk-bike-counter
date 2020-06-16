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
from ..video import OutputVideo, Video
from ..widgets.aoi import AOIDialog, AONIDialog, LOIDialog
from .base import BaseController


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
        self.input_camera_type = "file"
        self.camera_id = 0
        self.internal_webcam = IntVar()
        self.external_webcam = IntVar()
        self.stop_thread = BooleanVar()

    def open_file(self):
        file_name = filedialog.askopenfilename()
        if file_name:
            self.video = Video(file_name)
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
        tracker = ObjectTracker(self.mask)
        if self.video:
            tracker.video_filename = self.video.filename
        tracker.valid_selected_objects = [
            "pedestrian" if item == "person" else item
            for item in object_classes
            if item != "bicycle"
        ]
        tracker.object_classes = object_classes
        tracker.color_table = color_table
        tracker.video = self.video
        tracker.input_camera_type = self.input_camera_type
        tracker.camera_id = self.camera_id
        tracker.stop_thread = self.stop_thread
        tracker.output_video = self.output_video
        print(listener_object.handle_post_processed_frame)
        tracker.frame_listener = listener_object.handle_post_processed_frame
        tracker.track_objects(config)

    def add_new_aoi(self):
        if self.video:
            AOIDialog(self.view.parent, self.video.filename, self)
        else:
            messagebox.showwarning("Warning", "Please select a file!")

    def add_new_aoni(self):
        if self.video:
            AONIDialog(self.view.parent, self.video.filename, self)
        else:
            messagebox.showwarning("Warning", "Please select a file!")

    def add_new_loi(self):
        if self.video:
            LOIDialog(self.view.parent, self.video.filename, self)
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
        if self.listener_object:
            self.listener_object.stop_threads()
