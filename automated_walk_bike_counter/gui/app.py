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

import logging
import tkinter as tk

import cv2
import numpy as np
from PIL import Image

from ..core.configuration import config
from ..core.tracking.object_tracker import ObjectTracker
from .controller.main_controller import MainController
from .video import OutputVideo, Video
from .view.main import MainView


class Application:
    def __init__(self, root=None, *args, **kwargs):
        if root is None:
            root = tk.Tk()
            root.title("CSULA Object Detection Project")
        self.root = root

        self.controller = MainController()
        self.view = MainView(root, self.controller)

    def run(self):
        self.root.geometry("1280x720+100+100")
        self.root.mainloop()


class Cli:
    def __init__(self):
        self.valid_selected_objects = []
        self.aoi_points = config.aoi
        self.set_valid_selected_objects()

    def set_valid_selected_objects(self):
        allowed_objects = config.valid_objects
        for i in range(len(allowed_objects)):

            if allowed_objects[i] in config.search_objects:
                object_color = config.objects_colors[
                    config.search_objects.index(allowed_objects[i])
                ]
                c = object_color.lstrip("#")
                self.valid_selected_objects.append(
                    (
                        allowed_objects[i],
                        (tuple(int(c[i : i + 2], 16) for i in (0, 2, 4)), object_color),
                        1,
                    )
                )
            else:
                self.valid_selected_objects.append(
                    (allowed_objects[i], ((255, 255, 255), "#FFFFFF"), 0)
                )

    def get_objects_colors_list(self):

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

    def get_mask_image(self):
        video_width, video_height = self.get_input_video_dimension()
        mask_image = np.asarray(
            Image.new("RGB", (int(video_width), int(video_height)), 0)
        )
        return cv2.fillConvexPoly(
            mask_image, np.array(self.aoi_points, "int32"), (255, 255, 255), 8, 0
        )

    def get_input_video_dimension(self):
        width, height = 0, 0
        if config.file_name != "":
            vcap = cv2.VideoCapture(config.file_name)
            if vcap.isOpened():
                width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                vcap.release()
        return width, height

    def run(self):
        output_video = None
        object_classes, color_table = self.get_objects_colors_list()
        print(f"Tracked objects: {str(color_table)}")
        mask = []
        if self.aoi_points and len(self.aoi_points) > 2:
            mask = self.get_mask_image()
        tracker = ObjectTracker(mask)

        if config.file_name != "":
            tracker.video_filename = config.file_name
            video = Video(config.file_name)
            output_video = OutputVideo(video)
            if self.aoi_points:
                output_video.has_AOI = True
            tracker.video = video
            tracker.output_video = output_video

        tracker.valid_selected_objects = [
            "pedestrian" if item == "person" else item
            for item in object_classes
            if item != "bicycle"
        ]

        tracker.object_classes = object_classes
        tracker.color_table = color_table
        tracker.input_camera_type = config.input_type
        tracker.camera_id = config.camera_id
        tracker.track_objects(config)


def main():
    logging.basicConfig(level=config.log)

    if config.cli:
        if config.input_type == "file":
            print(f"Running with file name {config.file_name}")
        else:
            print(f"Running with camera ID: {config.camera_id}")
        cli = Cli()
        cli.run()
    else:
        app = Application()
        app.run()
