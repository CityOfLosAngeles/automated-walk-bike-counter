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
import math
from tkinter import DISABLED, HORIZONTAL, E, Label, LabelFrame, StringVar, W
from tkinter.ttk import Combobox, Scale


class SettingsPane(LabelFrame):
    def __init__(self, parent, controller):
        super(SettingsPane, self).__init__(master=parent, text="Settings")
        self.controller = controller
        self.res_combo = Combobox(master=self)
        self.aoi_combo = Combobox(master=self, values=("Not Specified", "Specified"))
        self.aoi_output_present_combo = Combobox(master=self, values=("No", "Yes"))
        self.scale_value = StringVar()
        self.aoi_opaque = Scale(
            master=self,
            from_=0,
            to=30,
            orient=HORIZONTAL,
            command=self.set_output_video_opaque,
        )
        self.label4 = Label(master=self, text="AOI opaque (%) : ")

        self.controller.video_settings_pane = self

        self.initialize_resolution_combo()

    def initialize_resolution_combo(self):
        predef_sizes = [320, 416, 608]
        if self.controller.video:
            if (
                self.controller.video.width > predef_sizes[-1]
                and self.controller.video.height > predef_sizes[-1]
            ):
                predef_sizes.append(self.controller.video.height)
                predef_sizes.append(self.controller.video.width)

        label1 = Label(master=self, text="Resolution : ")
        self.res_combo["values"] = predef_sizes
        self.res_combo.current(1)
        self.res_combo.bind("<<ComboboxSelected>>", self.select_the_resolution)
        self.select_the_resolution(None)

        label2 = Label(master=self, text="AOI :")

        self.aoi_combo.current(int(self.controller.output_video.has_AOI))
        self.aoi_combo.state(["disabled"])

        label3 = Label(master=self, text="Show AOI in output :")

        self.aoi_output_present_combo.config(state=DISABLED)

        self.aoi_output_present_combo.current(0)
        self.aoi_output_present_combo.config(state=DISABLED)
        self.aoi_output_present_combo.bind(
            "<<ComboboxSelected>>", self.aoi_output_present_change
        )

        test_label = Label(master=self, textvariable=self.scale_value)

        self.aoi_opaque.state(["disabled"])

        label1.grid(row=0, column=0, padx=3, pady=3, sticky=W)
        self.res_combo.grid(row=0, column=1, padx=3, pady=3, sticky=[E, W])
        label2.grid(row=1, column=0, padx=3, pady=3, sticky=W)
        self.aoi_combo.grid(row=1, column=1, padx=3, pady=3, sticky=[E, W])
        label3.grid(row=2, column=0, padx=3, pady=3, sticky=W)
        self.aoi_output_present_combo.grid(
            row=2, column=1, padx=3, pady=3, sticky=[E, W]
        )
        self.label4.grid(row=3, column=0, padx=3, pady=3, sticky=W)
        test_label.grid(row=3, column=0, padx=3, pady=3, sticky=E)
        self.aoi_opaque.grid(row=3, column=1, padx=3, pady=3, sticky=[E, W])

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def select_the_resolution(self, event):
        self.controller.output_video.resolution = self.res_combo.get()

    def update_aoi_changes(self):
        if self.controller.output_video.has_AOI:
            self.aoi_output_present_combo.state(["!disabled"])
            self.aoi_opaque.state(["!disabled"])
            self.controller.output_video.opaque = 10
            self.aoi_opaque.set(10)
        else:
            self.aoi_output_present_combo.state(["disabled"])
            self.controller.output_video.opaque = 0
            self.aoi_opaque.set(0)
            self.aoi_opaque.state(["disabled"])

        self.aoi_output_present_combo.current(int(self.controller.output_video.has_AOI))
        self.controller.output_video.AOI_output_present = (
            self.controller.output_video.has_AOI
        )
        self.aoi_combo.current(int(self.controller.output_video.has_AOI))

    def aoi_output_present_change(self, event):
        self.controller.output_video.AOI_output_present = bool(
            self.aoi_output_present_combo.current()
        )

    def set_output_video_opaque(self, value):
        self.controller.output_video.opaque = math.floor(float(value))
        self.scale_value.set(math.floor(float(value)))
        logging.debug(f"New opacity: {str(self.controller.output_video.opaque)}")
