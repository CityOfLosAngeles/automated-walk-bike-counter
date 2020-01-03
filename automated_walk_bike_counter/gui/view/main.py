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

import os
from tkinter import Button, E, Frame, N, W, messagebox

from ..widgets.left_frame import Left_Frame
from ..widgets.menu import App_Menu
from ..widgets.video_frame import Video_Frame
from .base import BaseView


class MainView(BaseView):
    def __init__(self, parent, controller):

        super(MainView, self).__init__(parent, controller)
        self.object_detection_thread = None
        self.loading_thread = None
        self.video_player_canvas = None
        self.canvas_image = None
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.color_objects = []
        self.checkbox_variables = []
        self.view_initializer()
        # array to hold list of area of interests
        self.area_of_interest = []
        self.video_player_widget = None
        self.controller = controller

    def view_initializer(self):

        self.initialize_menu()

        top_frame = Frame(self.parent, height=50, pady=5)
        middle_frame = Frame(self.parent, height=600)
        bottom_frame = Frame(self.parent, height=70, pady=5)

        top_frame.grid(row=0, column=0, sticky=E + W)
        middle_frame.grid(row=1, sticky=E + W)
        bottom_frame.grid(row=2, sticky=E + W)

        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_rowconfigure(2, weight=1)

        self.left_frame = Left_Frame(middle_frame, self.controller)
        self.video_frame = Video_Frame(middle_frame, self.controller)

        self.left_frame.grid(row=0, column=0, sticky=W + E + N)
        self.video_frame.grid(row=0, column=1)

        middle_frame.grid_columnconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(1, weight=1)

        self.initialize_top_frame(top_frame)

    def initialize_menu(self):
        App_Menu(self.parent, self.controller)

    def initialize_top_frame(self, parent):
        frame = Frame(master = parent, width=1220)
        frame.grid(row=0,column=0,sticky=E+W)
        place_holder = Frame(master=frame, width=1140)
        place_holder.grid(row=0, column=0)
        button = Button(master = frame, text= "Generate")
        button.bind("<ButtonPress-1>", self.generate_button_click)
        button.grid(row=0, column=1)

        stop_button = Button(master = frame, text= "Stop")
        stop_button.bind("<ButtonPress-1>", self.cancel_button_click)
        stop_button.grid(row=0, column=2)

        self.parent.grid_columnconfigure(0,weight=1)
        self.parent.grid_columnconfigure(1, weight=0)
        self.parent.grid_columnconfigure(2, weight=0)

    def generate_button_click(self, event):
        if self.controller.input_camera_type == "file":
            if not self.controller.video:
                messagebox.showwarning("Warning", "Please select a file!")
            else:
                self.video_frame.initialize_canvas()
        else:
            self.video_frame.initialize_canvas()

    def update_setting_aoi_status(self):
        self.left_frame.settings_pane.update_aoi_changes()

    def cancel_button_click(self, event):
        self.controller.cancel_tracking_process()
