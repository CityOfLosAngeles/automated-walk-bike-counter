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
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    E,
    Entry,
    Frame,
    Label,
    N,
    S,
    StringVar,
    Toplevel,
    W,
    messagebox,
)

import cv2
import numpy as np
from PIL import Image, ImageTk

from ...model.video.LOI_info import LOIInfo


class AOIDialog(object):
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
        self.video_width = self.controller.video.width
        self.video_height = self.controller.video.height
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
        self.label2 = Label(
            top,
            text=(
                "Hint: to add a new vertex left click and to close the polygon right"
                + " click"
            ),
        )
        self.get_areas_name_frame = Frame(top, width=400)
        video_new_height = self.get_video_resized_dimensions()[1]
        self.video_frame = Frame(top, width=800, height=600)
        self.canvas = self.initialize_canvas(video_new_height)
        self.canvas.grid(row=0, column=0, sticky=(W, E))
        self.video_frame.rowconfigure(0, weight=1)
        buttons_frame = Frame(top)

        self.label1.grid(row=0, column=1, columnspan=2, sticky=(W, N))
        self.label2.grid(row=1, column=1, columnspan=2, sticky=(W, N), pady=(5, 0))
        self.get_areas_name_frame.grid(
            row=0, column=3, rowspan=2, columnspan=4, sticky=(W, N)
        )
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
        self.polygon_A_label = None
        self.polygon_B_label = None
        self.polygon_A_center = None
        self.polygon_B_center = None
        self.loi_identified = False
        self.frame_areas_image = []
        self.has_named_areas = None
        self.are_areas_named = BooleanVar()
        self.area1_name = StringVar()
        self.area2_name = StringVar()
        self.assign_name_frame = None
        self.start_point = []
        self.end_point = []
        self.side_a_name = None
        self.side_b_name = None
        self.resized_height = None
        AOIDialog.__init__(self, parent, filename, controller)

    def initialize_subclass_components(self):
        self.area1_name.set("Side 1")
        self.area2_name.set("Side 2")
        self.resized_height = int(self.get_video_resized_dimensions()[1])
        self.has_named_areas = None
        self.has_named_areas = Checkbutton(
            master=self.get_areas_name_frame,
            text="Named areas",
            variable=self.are_areas_named,
            command=self.has_named_areas_clicked,
            state=DISABLED,
        )
        self.has_named_areas.grid(row=0, column=1, columnspan=5, sticky=W)

        self.assign_name_frame = Frame(self.get_areas_name_frame)
        self.assign_name_frame.grid(row=1, column=1, columnspan=5, sticky=(W, E))

        self.assign_name_frame.grid_remove()

        self.top.title("Specifying the line of interest")
        self.label1.config(
            text=(
                "Please select your line of interest (must be just a line) on your "
                "video:"
            )
        )
        self.label2.config(
            text="Hint: to add a line just specify two pints of it by left click"
        )
        self.btn_save.config(text="Save LOI")
        self.btn_delete.config(text="Delete LOI", command=self.delete_loi)

        self.frame_areas_image = np.asarray(
            Image.new("RGB", (self.video_frame_width, self.video_frame_height), 0)
        )

        if self.controller.video.line_of_interest_info is None:
            self.mask_image = self.get_empty_mask_image()
        else:
            if len(self.controller.video.line_of_interest_info.preview_points) > 0:
                self.points = self.controller.video.line_of_interest_info.preview_points
                self.draw_line_splited_areas(
                    self.points[0][0],
                    self.points[0][1],
                    self.points[1][0],
                    self.points[1][1],
                )
                self.btn_delete.config(state=NORMAL)
            if self.controller.video.line_of_interest_info is not None:
                self.area1_name.set(
                    self.controller.video.line_of_interest_info.side_A_name
                )
                self.area2_name.set(
                    self.controller.video.line_of_interest_info.side_B_name
                )

    def on_mouse_right_click(self, event):
        AOIDialog.on_mouse_right_click(self, event)

    def on_mouse_click(self, event):
        x = event.x
        y = event.y
        if not self.loi_identified:
            if self.drawing:
                x0 = self.points[0][0]
                y0 = self.points[0][1]

                if x0 == x and y0 == y:
                    messagebox.showerror(
                        "Error",
                        "Start point and end point can not be the same.",
                        parent=self.top,
                    )
                    self.clear()
                else:
                    self.draw_line_splited_areas(x0, y0, x, y)

                self.drawing = False
            else:
                if len(self.points) > 0:
                    message_box = messagebox.askquestion(
                        "Warning",
                        "You are drawing a new line. Are you sure?",
                        icon="warning",
                        parent=self.top,
                    )
                    if message_box == "yes":
                        self.clear()
                    else:
                        return

                self.drawing = True
            self.points.append((x, y))

    def draw_line_splited_areas(self, px1, py1, px2, py2):

        if px1 != px2:
            slope = (py2 - py1) / (px2 - px1)

        image_edge_intersects = []
        y1 = 0
        if py2 != py1:
            if px1 == px2:
                top_horizontal_intersect_x = px1
            else:
                top_horizontal_intersect_x = (y1 - py1) / slope + px1

            if 0 <= top_horizontal_intersect_x <= self.video_frame_width:
                image_edge_intersects.append((top_horizontal_intersect_x, y1))

        x1 = self.video_frame_width
        if px1 != px2:
            right_vertical_intersect_y = slope * x1 - slope * px1 + py1
            if 0 <= right_vertical_intersect_y <= self.video_frame_height:
                image_edge_intersects.append((x1, right_vertical_intersect_y))

        y1 = self.video_frame_height
        if py2 != py1:
            if px1 == px2:
                bottom_horizontal_intersect_x = px1
            else:
                bottom_horizontal_intersect_x = (y1 - py1) / slope + px1

            if 0 <= bottom_horizontal_intersect_x <= self.video_frame_width:
                image_edge_intersects.append((bottom_horizontal_intersect_x, y1))

        x1 = 0
        if px1 != px2:
            left_vertical_intersect_y = slope * x1 - slope * px1 + py1
            if 0 <= left_vertical_intersect_y <= self.video_frame_height:
                image_edge_intersects.append((x1, left_vertical_intersect_y))

        start_point = (
            int(image_edge_intersects[0][0]),
            int(image_edge_intersects[0][1]),
        )
        self.ori_points.append(
            (
                int(self.get_convert_to_original_coefficient() * start_point[0]),
                int(self.get_convert_to_original_coefficient() * start_point[1]),
            )
        )
        end_point = (int(image_edge_intersects[1][0]), int(image_edge_intersects[1][1]))
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

        area_1_points = self.get_line_borders_intersection_points(
            start_point[0], start_point[1], end_point[0], end_point[1]
        )

        self.draw_polygon_on_areas_image("polygon_A", area_1_points)

        self.ori_points = self.convert_points_dimensions_to_original(area_1_points)

        self.mask_image = cv2.fillConvexPoly(
            self.mask_image, np.array(self.ori_points, "int32"), (255, 255, 255), 8, 0
        )

        self.start_point = start_point
        self.end_point = end_point
        self.loi_identified = True
        self.has_named_areas.config(state=NORMAL)

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
            line_of_interest_info = LOIInfo()
            line_of_interest_info.mask_image = self.mask_image
            line_of_interest_info.points = self.line_of_interest_points
            line_of_interest_info.preview_points = self.points
            if self.are_areas_named.get():
                line_of_interest_info.sides_have_name = True
                line_of_interest_info.side_A_name = self.side_a_name.get()
                line_of_interest_info.side_B_name = self.side_b_name.get()
            else:
                if line_of_interest_info.sides_have_name:
                    line_of_interest_info.sides_have_name = False
                    line_of_interest_info.side_A_name = None
                    line_of_interest_info.side_B_name = None

            self.controller.video.line_of_interest_info = line_of_interest_info

            self.close_window()
        else:
            messagebox.showwarning(
                "Warning", "You have not determined a valid LOI!", parent=self.top
            )

    def delete_loi(self):
        self.controller.video.line_of_interest_info = None
        self.btn_delete.config(state=DISABLED)
        self.close_window()

    def clear(self):
        AOIDialog.clear(self)
        self.loi_identified = False
        self.frame_areas_image = np.asarray(
            Image.new("RGB", (self.video_frame_width, self.video_frame_height), 0)
        )
        self.points = []
        self.line_of_interest_points = []

        if self.canvas.find_withtag("polygon_A"):
            self.canvas.delete(self.canvas.find_withtag("polygon_A"))
        if self.canvas.find_withtag("polygon_B"):
            self.canvas.delete(self.canvas.find_withtag("polygon_B"))

        self.are_areas_named.set(False)
        self.has_named_areas.config(state=DISABLED)
        self.assign_name_frame.grid_remove()

    def get_line_borders_intersection_points(
        self, start_point_x, start_point_y, end_point_x, end_point_y
    ):
        # This function returns the list of points which is the collection of
        # intersections of the line and the image borders and also the coordinates
        # of the other corners of the image that can be used to compose one of the
        # polygons that are created by splitting the image through the line

        is_start_point = True

        polygon_points = [[start_point_x, start_point_y]]

        while (
            start_point_x != end_point_x
            and start_point_y != end_point_y
            or is_start_point
        ):
            if start_point_x == 0 and start_point_y != 0:
                start_point_x, start_point_y = 0, 0
            elif start_point_y == 0 and start_point_x != self.video_frame_width:
                start_point_x, start_point_y = self.video_frame_width, 0
            elif (
                start_point_x == self.video_frame_width
                and start_point_y != self.resized_height
            ):
                start_point_x, start_point_y = (
                    self.video_frame_width,
                    self.resized_height,
                )
            elif start_point_y == self.resized_height:
                start_point_x, start_point_y = 0, self.resized_height

            polygon_points.append((start_point_x, start_point_y))

            if is_start_point:
                is_start_point = False

        polygon_points.append([end_point_x, end_point_y])

        return polygon_points

    def convert_points_dimensions_to_original(self, points):
        new_points = []
        for (x, y) in points:

            x = int(self.get_convert_to_original_coefficient() * x)
            y = int(self.get_convert_to_original_coefficient() * y)
            new_points.append((x, y))

        return new_points

    def draw_polygon_on_areas_image(self, area_name, points):

        if area_name == "polygon_A":
            cv2.fillConvexPoly(
                self.frame_areas_image, np.array(points, "int32"), (255, 255, 255), 8, 0
            )

    def has_named_areas_clicked(self):
        if not self.are_areas_named.get():
            self.assign_name_frame.grid_remove()
            self.canvas.delete(self.canvas.find_withtag("polygon_A"))
            self.canvas.delete(self.canvas.find_withtag("polygon_B"))

        else:
            if self.loi_identified:
                self.assign_name_frame.grid()

                Label(master=self.assign_name_frame, text="Side-1 name:").grid(
                    row=0, column=1, sticky=W
                )

                validate_entry_input = (
                    self.assign_name_frame.register(self.validate_entry),
                    "%W",
                    "%P",
                    "%S",
                )

                self.side_a_name = Entry(
                    master=self.assign_name_frame,
                    textvariable=self.area1_name,
                    validatecommand=validate_entry_input,
                    validate="key",
                )
                self.side_a_name.grid(row=0, column=2)
                self.side_a_name.bind(
                    "<KeyRelease>",
                    lambda event, args="SideA": self.areas_text_name_change(
                        event, args
                    ),
                )
                # self.side_a_name.insert(END, self.area1_name.get())

                Label(master=self.assign_name_frame, text="Side-2 name:").grid(
                    row=0, column=4
                )

                self.side_b_name = Entry(
                    master=self.assign_name_frame,
                    textvariable=self.area2_name,
                    validatecommand=validate_entry_input,
                    validate="key",
                )

                self.side_b_name.grid(row=0, column=5)
                self.side_b_name.bind(
                    "<KeyRelease>",
                    lambda event, args="SideB": self.areas_text_name_change(
                        event, args
                    ),
                )
                # self.side_b_name.insert(END, self.area2_name.get())

                self.polygon_A_center = self.find_centroid(
                    self.get_line_borders_intersection_points(
                        self.start_point[0],
                        self.start_point[1],
                        self.end_point[0],
                        self.end_point[1],
                    )
                )
                self.polygon_B_center = self.find_centroid(
                    self.get_line_borders_intersection_points(
                        self.end_point[0],
                        self.end_point[1],
                        self.start_point[0],
                        self.start_point[1],
                    )
                )

                self.polygon_A_label = self.canvas.create_text(
                    self.polygon_A_center[0],
                    self.polygon_A_center[1],
                    text=self.area1_name.get(),
                    font=("", 36),
                    fill="#FFFFFF",
                    tag="polygon_A",
                )

                self.polygon_B_label = self.canvas.create_text(
                    self.polygon_B_center[0],
                    self.polygon_B_center[1],
                    text=self.area2_name.get(),
                    font=("", 36),
                    fill="#FFFFFF",
                    tag="polygon_B",
                )

    def find_centroid(self, points):
        ans = [0, 0]

        n = len(points)
        signed_area = 0

        # For all vertices
        for i in range(len(points)):
            x0 = points[i][0]
            y0 = points[i][1]
            x1 = points[(i + 1) % n][0]
            y1 = points[(i + 1) % n][1]

            # Calculate value of A
            # using shoelace formula
            A = (x0 * y1) - (x1 * y0)
            signed_area += A

            # Calculating coordinates of
            # centroid of polygon
            ans[0] += (x0 + x1) * A
            ans[1] += (y0 + y1) * A

        signed_area *= 0.5
        ans[0] = (ans[0]) / (6 * signed_area)
        ans[1] = (ans[1]) / (6 * signed_area)

        return ans

    def areas_text_name_change(self, event, name):
        if name == "SideA":
            self.canvas.itemconfig(
                # self.polygon_A_label, text=self.side_a_name.get() + event.char
                self.polygon_A_label,
                text=self.area1_name.get(),
            )
        else:
            self.canvas.itemconfig(
                # self.polygon_B_label, text=self.side_b_name.get() + event.char
                self.polygon_B_label,
                text=self.area2_name.get(),
            )

    def validate_entry(self, widget_name, prev_value, inserted_text):
        if len(prev_value) > 20:
            messagebox.showerror(
                "Error",
                "The length of the name shouldn't be greater than 20.",
                parent=self.assign_name_frame,
            )
            return False
        else:
            return True
