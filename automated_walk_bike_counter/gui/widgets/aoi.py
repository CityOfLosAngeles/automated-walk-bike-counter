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

import threading
from tkinter import (
    CENTER,
    DISABLED,
    NORMAL,
    Button,
    Canvas,
    E,
    Frame,
    Label,
    N,
    S,
    Toplevel,
    W,
)

import cv2
import numpy as np
from PIL import Image, ImageTk


class AOIDialog:
    def __init__(self, parent, filename, controller):
        top = self.top = Toplevel(parent)
        self.filename = filename
        self.controller = controller
        self.video_frame_width = 800
        self.video_frame_height = 600
        self.thread_event = None
        self.video_loading_thread = None
        self.drawing = False
        self.points = []
        self.ori_points = []
        camera = cv2.VideoCapture(self.filename)
        self.video_width = int(camera.get(3))
        self.video_height = int(camera.get(4))
        self.empty_image = Image.new("RGB", (self.video_width, self.video_height), 0)
        self.mask_image = self.get_empty_mask_image()
        self.btn_delete = None
        self.initialize_video_thread()

        self.label1 = Label(
            top,
            text=(
                "Please select your area of interest (must be a polygon) on your video:"
            ),
        )
        label2 = Label(
            top,
            text=(
                "Hint: to add a new vertex left click and to close the polygon right"
                + "click"
            ),
        )
        video_new_height = self.get_video_resized_dimensions()[1]
        self.video_frame = Frame(top, bg="green", width=800, height=600)
        self.canvas = self.initialize_canvas(video_new_height)
        self.canvas.grid(row=0, column=0, sticky=(W, E))
        self.video_frame.rowconfigure(0, weight=1)

        buttons_frame = Frame(top)

        self.label1.grid(row=0, column=1, columnspan=6, sticky=(W, N))
        label2.grid(row=1, column=1, columnspan=6, sticky=(W, N))
        self.video_frame.grid(row=2, column=1, columnspan=4, sticky=(W, E))
        buttons_frame.grid(row=3, column=1, columnspan=4, sticky=(W, E, N, S))

        self.btn_save = Button(
            buttons_frame, text="Save AOI", width=20, command=self.save_mask
        )
        btn_clear = Button(buttons_frame, text="Clear", width=20, command=self.clear)
        btn_cancel = Button(
            buttons_frame, text="Cancel", width=20, command=self.close_window
        )
        self.btn_delete = Button(
            buttons_frame, text="Delete AOI", width=20, command=self.delete_aoi
        )
        if self.controller.output_video.has_AOI:
            self.btn_delete.config(state=NORMAL)
        else:
            self.btn_delete.config(state=DISABLED)

        self.btn_save.grid(row=0, column=1, padx=(0, 10), pady=(5, 20), sticky=(W, E))
        btn_clear.grid(row=0, column=2, padx=(10, 10), pady=(5, 20), sticky=(W, E))
        btn_cancel.grid(row=0, column=3, padx=(10, 10), pady=(5, 20), sticky=(W, E))
        self.btn_delete.grid(row=0, column=4, padx=(10, 0), pady=(5, 20), sticky=(W, E))

        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=0)
        buttons_frame.columnconfigure(2, weight=0)
        buttons_frame.columnconfigure(3, weight=0)
        buttons_frame.columnconfigure(4, weight=0)
        buttons_frame.columnconfigure(5, weight=1)

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=0)
        top.columnconfigure(3, weight=0)
        top.columnconfigure(4, weight=0)
        top.columnconfigure(5, weight=1)
        top.rowconfigure(2, weight=1)

        self.initialize_subclass_components()
        top.protocol("WM_DELETE_WINDOW", self.close_window)
        top.geometry("850x630+100+100")

        top.focus_set()
        top.grab_set()
        top.wait_window()

    def initialize_canvas(self, video_resized_height):

        canvas = Canvas(self.video_frame, width=800, height=video_resized_height)
        canvas.bind("<ButtonPress-1>", self.on_mouse_click)
        canvas.bind("<ButtonPress-3>", self.on_mouse_right_click)
        canvas.bind("<Motion>", self.on_mouse_move_callback)

        return canvas

    def initialize_video_thread(self):

        self.thread_event = threading.Event()
        self.video_loading_thread = threading.Thread(
            target=self.play_bg_video, args=(self.thread_event,), daemon=True
        )
        self.video_loading_thread.start()

    def save_mask(self):

        if (
            not (
                np.array_equal(np.asarray(self.mask_image), self.get_empty_mask_image())
            )
            and len(self.points) > 2
        ):
            self.save_result_to_parent(self.mask_image)
            self.controller.output_video.has_AOI = True
            self.controller.refresh_aoi_status()
            self.close_window()
        else:
            self.top.messagebox.showwarning(
                "Warning", "You have not determined a valid AOI!"
            )

    def close_window(self):

        self.thread_event.set()
        while not self.thread_event.is_set():
            self.thread_event.wait()
        self.top.destroy()

    def delete_aoi(self):
        self.controller.mask = None
        self.controller.output_video.has_AOI = False
        self.btn_delete.config(state=DISABLED)
        self.controller.refresh_aoi_status()
        self.close_window()

    def play_bg_video(self, e):

        camera = cv2.VideoCapture(self.filename)

        while camera.isOpened() and not e.is_set():
            ret, frame = camera.read()

            if ret:

                dims = self.get_video_resized_dimensions()

                frame = cv2.resize(frame, dims, interpolation=cv2.INTER_AREA)

                frame_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_image)
                frame_image = ImageTk.PhotoImage(frame_image)

                if not e.is_set():
                    self.canvas.create_image(
                        dims[0] / 2 + 2,
                        dims[1] / 2 + 2,
                        anchor=CENTER,
                        image=frame_image,
                        tag="canvas_image",
                    )
                    self.canvas.tag_lower("canvas_image")
                    self.canvas.image = frame_image

    def on_mouse_click(self, event):
        x = event.x
        y = event.y
        ori_x = int(self.get_convert_to_original_coefficient() * x)
        ori_y = int(self.get_convert_to_original_coefficient() * y)

        if self.drawing:
            self.canvas.create_line(
                self.points[-1][0],
                self.points[-1][1],
                x,
                y,
                fill="#ffffff",
                width=2,
                tag="foreground",
            )
            cv2.line(
                self.mask_image,
                (self.ori_points[-1][0], self.ori_points[-1][1]),
                (ori_x, ori_y),
                (255, 255, 255),
                2,
            )
        else:
            self.drawing = True

        self.points.append((x, y))
        self.ori_points.append((ori_x, ori_y))

    def on_mouse_right_click(self, event):
        if len(self.points) > 1:
            self.canvas.create_line(
                self.points[-1][0],
                self.points[-1][1],
                self.points[0][0],
                self.points[0][1],
                fill="#ffffff",
                width=2,
                tag="foreground",
            )
            cv2.line(
                self.mask_image,
                (self.ori_points[-1][0], self.ori_points[-1][1]),
                (self.ori_points[0][0], self.ori_points[0][1]),
                (255, 255, 255),
                2,
            )
            self.canvas.delete(self.canvas.find_withtag("line"))

        self.mask_image = cv2.fillConvexPoly(
            self.mask_image, np.array(self.ori_points, "int32"), (255, 255, 255), 8, 0
        )
        self.drawing = False

    def on_mouse_move_callback(self, event):
        x = event.x
        y = event.y

        if len(self.points) > 0 and self.drawing:
            self.canvas.delete(self.canvas.find_withtag("line"))
            self.canvas.create_line(
                self.points[-1][0],
                self.points[-1][1],
                x,
                y,
                fill="#ffffff",
                tag="line",
                width=2,
                dash=(4, 2),
            )

    def get_video_resized_dimensions(self):

        r = 1 / self.get_convert_to_original_coefficient()
        return 800, int(self.video_height * r)

    def get_convert_to_original_coefficient(self):
        return float(self.video_width) / 800

    def clear(self):
        self.drawing = False
        self.points = []
        self.ori_points = []
        self.canvas.delete("foreground", "line")
        self.mask_image = self.get_empty_mask_image()

    def save_result_to_parent(self, mask_image):
        self.controller.mask = mask_image

    def get_empty_mask_image(self):
        img = Image.new("RGB", (self.video_width, self.video_height), 0)
        return np.asarray(img)

    def initialize_subclass_components(self):
        self.top.title("Selecting the area of interest")


class AONIDialog(AOIDialog):
    def __init__(self, parent, filename, controller):
        AOIDialog.__init__(self, parent, filename, controller)

    def initialize_subclass_components(self):
        self.top.title("Selecting the area of non-interest")
        self.label1.config(
            text=(
                "Please select your area of non-interest (must be a polygon) on your "
                "video:"
            )
        )
        self.btn_save.config(text="Save AONI")
        self.btn_delete.config(text="Delete AONI")

        if len(self.controller.video.area_of_not_interest_mask) == 0:
            self.mask_image = self.get_empty_mask_image()
        else:
            self.mask_image = self.controller.video.area_of_not_interest_mask

    def on_mouse_right_click(self, event):
        AOIDialog.on_mouse_right_click(self, event)

    def save_mask(self):

        if (
            not (
                np.array_equal(np.asarray(self.mask_image), self.get_empty_mask_image())
            )
            and len(self.points) > 2
        ):
            self.controller.video.area_of_not_interest_mask = self.mask_image
            self.close_window()
        else:
            self.top.messagebox.showwarning(
                "Warning", "You have not determined a valid AONI!"
            )


class LOIDialog(AOIDialog):
    def __init__(self, parent, filename, controller):
        self.line_of_interest_points = []
        AOIDialog.__init__(self, parent, filename, controller)

    def initialize_subclass_components(self):
        self.btn_save.config(text="Save LOI")
        self.btn_delete.config(text="Delete LOI", command=self.delete_loi)

        if len(self.controller.video.line_of_interest_points) == 0:
            self.mask_image = self.get_empty_mask_image()
        else:
            self.points = self.controller.video.line_of_interest_points
            self.draw_line_splited_areas(
                self.points[0][0],
                self.points[0][1],
                self.points[1][0],
                self.points[1][1],
            )
            self.btn_delete.config(state=NORMAL)

    def on_mouse_right_click(self, event):
        AOIDialog.on_mouse_right_click(self, event)

    def on_mouse_click(self, event):
        x = event.x
        y = event.y

        if self.drawing:
            x0 = self.points[0][0]
            y0 = self.points[0][1]

            self.draw_line_splited_areas(x0, y0, x, y)

            self.drawing = False
        else:
            self.drawing = True

        self.points.append((x, y))

    def draw_line_splited_areas(self, px1, py1, px2, py2):
        slope = (py2 - py1) / (px2 - px1)

        # Line equation is : ax + by + c = 0
        # slope = (y - y1)/(x - x1)
        # y = slope*x - slope*x1 + y1
        # slope * x - y - (slope * x1) + y1 = 0

        image_edge_intersects = []

        y1 = 0
        top_horizontal_intersect_x = (y1 - py1) / slope + px1
        if 0 <= top_horizontal_intersect_x <= self.video_frame_width:
            image_edge_intersects.append((top_horizontal_intersect_x, y1))

        x1 = self.video_frame_width
        right_vertical_intersect_y = slope * x1 - slope * px1 + py1
        if 0 <= right_vertical_intersect_y <= self.video_frame_height:
            image_edge_intersects.append((x1, right_vertical_intersect_y))

        y1 = self.video_frame_height
        bottom_horizontal_intersect_x = (y1 - py1) / slope + px1
        if 0 <= bottom_horizontal_intersect_x <= self.video_frame_width:
            image_edge_intersects.append((bottom_horizontal_intersect_x, y1))

        x1 = 0
        left_vertical_intersect_y = slope * x1 - slope * px1 + py1
        if 0 <= left_vertical_intersect_y <= self.video_frame_height:
            image_edge_intersects.append((x1, left_vertical_intersect_y))

        start_point = (image_edge_intersects[0][0], image_edge_intersects[0][1])
        self.ori_points.append(
            (
                int(self.get_convert_to_original_coefficient() * start_point[0]),
                int(self.get_convert_to_original_coefficient() * start_point[1]),
            )
        )
        end_point = (image_edge_intersects[1][0], image_edge_intersects[1][1])
        self.draw_line(start_point[0], start_point[1], end_point[0], end_point[1])

        self.line_of_interest_points.append(
            (
                int(self.get_convert_to_original_coefficient() * start_point[0]),
                int(self.get_convert_to_original_coefficient() * start_point[1]),
            )
        )
        self.line_of_interest_points.append(
            (
                int(self.get_convert_to_original_coefficient() * end_point[0]),
                int(self.get_convert_to_original_coefficient() * end_point[1]),
            )
        )

        while start_point[0] != end_point[0] and start_point[1] != end_point[1]:
            if start_point[0] == 0:
                start_point = (0, 0)
            elif start_point[1] == 0 and start_point[0] != self.video_frame_width:
                start_point = (self.video_frame_width, 0)
            elif (
                start_point[0] == self.video_frame_width
                and start_point[1] != self.video_frame_height
            ):
                start_point = (self.video_frame_width, self.video_frame_height)
            elif start_point[1] == self.video_frame_height:
                start_point = (0, self.video_frame_height)
            self.ori_points.append(
                (
                    int(self.get_convert_to_original_coefficient() * start_point[0]),
                    int(self.get_convert_to_original_coefficient() * start_point[1]),
                )
            )

        self.draw_line(start_point[0], start_point[1], end_point[0], end_point[1])
        self.ori_points.append(
            (
                int(self.get_convert_to_original_coefficient() * end_point[0]),
                int(self.get_convert_to_original_coefficient() * end_point[1]),
            )
        )

        self.mask_image = cv2.fillConvexPoly(
            self.mask_image, np.array(self.ori_points, "int32"), (255, 255, 255), 8, 0
        )

    def draw_line(self, px1, py1, px2, py2):

        self.canvas.create_line(
            px1, py1, px2, py2, fill="#FF0000", width=4, tag="foreground",
        )

    def save_mask(self):

        if (
            not (
                np.array_equal(np.asarray(self.mask_image), self.get_empty_mask_image())
            )
            and len(self.points) > 1
        ):
            self.controller.video.line_of_interest_mask = self.mask_image
            self.controller.video.line_of_interest_points = self.line_of_interest_points
            self.close_window()
        else:
            self.top.messagebox.showwarning(
                "Warning", "You have not determined a valid LOI!"
            )

    def delete_loi(self):
        self.controller.video.line_of_interest_mask = None
        self.controller.video.line_of_interest_points = []
        self.btn_delete.config(state=DISABLED)
        self.close_window()
