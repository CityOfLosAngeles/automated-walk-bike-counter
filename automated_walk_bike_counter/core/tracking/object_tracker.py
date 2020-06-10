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
import os
import re
from time import time as timer
from urllib.parse import urlparse

import cv2
import numpy as np
import tensorflow as tf
from munkres import Munkres

from ...utils.misc_utils import parse_anchors, read_class_names
from ...utils.nms_utils import gpu_nms
from ...utils.plot_utils import plot_one_box
from ..configuration import config
from ..frame import Frame
from ..model import YoloV3
from ..movingobject import MovingObject
from ..tracking.counter import ObjectCounter


class ObjectTracker:
    BOUNDRY = 30

    def __init__(self, mask_image):
        self.last_frame_moving_objects = []
        self.masked_image = mask_image
        self.moving_object_id_number = 0
        self.object_counter = ObjectCounter()
        self.current_frame = None
        self.current_frame_number = 0  # elapsed frames
        self.video_width = 0
        self.video_height = 0
        self.roiBasePoint = []
        self.video_filename = ""
        self.frame_listener = None
        self.video = None
        self.output_video = None
        self.color_table = {}
        self.image_processing_size = []
        self.object_classes = []
        self.object_costs = {
            "person": 100,
            "car": 200,
            "bus": 300,
            "truck": 200,
            "motorbike": 120,
            "bicycle": 100,
        }
        self.input_camera_type = ""
        self.camera_id = 0
        self.stop_thread = False
        self.background_frame = None
        self.periodic_counter_interval = 0
        self.valid_selected_objects = []

    def print_data_report_on_frame(self):
        if self.current_frame is not None:
            h, w = self.current_frame.postprocessed_frame.shape[:2]
            x = 5
            gap = 100
            y = h - 60
            if "person" in self.color_table:
                counter = "Ped:" + str(self.object_counter.COUNTER_p)
                cv2.putText(
                    self.current_frame.postprocessed_frame,
                    counter,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.color_table["person"],
                    2,
                )
                x = x + gap
            if "cyclist" in self.color_table:
                counter = "  Cyc:" + str(self.object_counter.COUNTER_c)
                cv2.putText(
                    self.current_frame.postprocessed_frame,
                    counter,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.color_table["cyclist"],
                    2,
                )
                x = x + gap
            if "car" in self.color_table:
                counter = "  Car:" + str(self.object_counter.COUNTER_car)
                cv2.putText(
                    self.current_frame.postprocessed_frame,
                    counter,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.color_table["car"],
                    2,
                )
                x = x + gap
            if "bus" in self.color_table:
                counter = "  Bus:" + str(self.object_counter.COUNTER_bus)
                cv2.putText(
                    self.current_frame.postprocessed_frame,
                    counter,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.color_table["bus"],
                    2,
                )
                x = x + gap
            if "truck" in self.color_table:
                counter = "  Truck:" + str(self.object_counter.COUNTER_truck)
                cv2.putText(
                    self.current_frame.postprocessed_frame,
                    counter,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.color_table["truck"],
                    2,
                )
                x = x + gap
            counter = "  Fr:" + str(self.current_frame_number)
            cv2.putText(
                self.current_frame.postprocessed_frame,
                counter,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def add_new_moving_object_to_counter(self, obj, position_new, postprocessed):
        self.object_counter.add_new_moving_object_for_counting(
            obj, position_new, postprocessed
        )

    def add_new_moving_object(self, current_detected_object):
        self.moving_object_id_number += 1
        new_moving_object = MovingObject(
            self.moving_object_id_number, current_detected_object.center
        )
        new_moving_object.last_detected_object = current_detected_object
        new_moving_object.add_position([current_detected_object.center])
        new_moving_object.init_kalman_filter()
        filtered_state_means, filtered_state_covariances = new_moving_object.kf.filter(
            new_moving_object.position
        )
        new_moving_object.set_next_mean(filtered_state_means[-1])
        new_moving_object.set_next_covariance(filtered_state_covariances[-1])
        new_moving_object.counted += 1

        # add to current_tracks
        self.last_frame_moving_objects.append(new_moving_object)

    def create_contour_for_current_objects(self, detected_objects):

        detected_objects_with_valid_contours = []

        for obj in detected_objects:
            (left, right, top, bot, mess, max_indx, confidence) = obj.box
            detected_objects_with_valid_contours.append(obj)

        return detected_objects_with_valid_contours

    def calculate_cost_matrix_for_moving_objects(self, current_detected_objects):
        def get_costs(current_object, cur_detected_objects):
            distances = []
            for obj in cur_detected_objects:

                dis = math.floor(
                    math.sqrt(
                        (obj.center_x - current_object.predicted_position[-1][0]) ** 2
                        + (obj.center_y - current_object.predicted_position[-1][1]) ** 2
                    )
                )
                dis += abs(
                    self.object_costs[current_object.last_detected_object.mess]
                    - self.object_costs[obj.mess]
                )
                distances.append(dis)
            return distances

        last_frame_moving_objects_cost_matrix = []
        valid_moving_objects = []
        for index, obj in enumerate(self.last_frame_moving_objects):
            # calculate costs for each tracked movingObjects using their predicted
            # position
            costs = get_costs(obj, current_detected_objects)

            # if moving object to all contours distances are too large, then not to
            # consider it at all
            threshold = config.ped_cost_threshold
            if obj.last_detected_object.mess == "bus":
                threshold = config.bus_cost_threshold
            elif obj.last_detected_object.mess == "truck":
                threshold = config.truck_cost_threshold

            if all(c > threshold for c in costs):
                # update it with KF predicted position
                obj.kalman_update_missing(obj.predicted_position[-1])
                # skip this moving object
                continue

            last_frame_moving_objects_cost_matrix.append(costs)
            # only valid moving objects are added to available_objecs
            valid_moving_objects.append(obj)

        return last_frame_moving_objects_cost_matrix, valid_moving_objects

    def update_skipped_frame(
        self, thresh,
    ):

        self.print_data_report_on_frame()

        self.remove_tracked_objects(thresh)
        print("Skipping frame due to no objects being tracked")

    def track_objects(self, args):

        anchors = parse_anchors(args.anchor_path)
        classes = read_class_names(args.class_name_path)
        self.object_counter.valid_selected_objects = self.valid_selected_objects
        if config.save_periodic_counter:
            self.object_counter.export_counter_initialization()

        num_class = len(classes)

        self.image_processing_size = args.new_size

        file = self.video_filename
        save_video = args.save_video

        # check if the video is reading from a file or from the webcam
        if self.input_camera_type == "webcam":
            file = self.camera_id
            vfname = "camera"
        else:
            # get the video name to process
            m = re.match(r"([^\.]*)(\..*)", file)
            vfname = m.group(1)
            vfname = m.string[: m.string.rfind(".")]
            if config.save_periodic_counter:
                self.object_counter.output_counter_file_name = vfname
                self.object_counter.export_counter_initialization()

            assert os.path.isfile(file), "file {} does not exist".format(file)

        camera = cv2.VideoCapture(file)

        if self.input_camera_type == "webcam":
            camera.set(3, 1200)
            camera.set(4, 800)

        self.video_width = int(camera.get(3))
        self.video_height = int(camera.get(4))

        assert camera.isOpened(), "Cannot capture source"

        if self.input_camera_type == "webcam":
            pass

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            outfile = vfname + "_result.mp4"
            print(f"Saving result to {outfile}")
            if self.input_camera_type == "webcam":
                # TODO: what is going on here??
                fps = 1
                if fps < 1:
                    fps = 1
            else:
                fps = round(camera.get(cv2.CAP_PROP_FPS))

            video_writer = cv2.VideoWriter(
                outfile, fourcc, fps, (self.video_width, self.video_height)
            )

        if config.save_periodic_counter and self.input_camera_type == "file":
            # Count the number of the frames that should pass in order to compute
            # the time for exporting the counter
            # Since the periodic_counter_time is in minutes we use the following
            # formula to compute the intervals
            # input counter time in minutes * video frame per second * number of
            # seconds in each min
            self.periodic_counter_interval = int(
                config.periodic_counter_time * self.video.fps * 60
            )

            # For testing purpose we can consider periodic_counter_time as seconds
            # and use the following formula
            # self.periodic_counter_interval =
            #           config.periodic_counter_time * self.video.fps

        start = timer()
        n = 0

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf.Session(config=cfg) as sess:

            input_data = tf.placeholder(
                tf.float32,
                [1, args.new_size[1], args.new_size[0], 3],
                name="input_data",
            )
            yolo_model = YoloV3(num_class, anchors)
            with tf.variable_scope("yolov3"):
                pred_feature_maps = yolo_model.forward(input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            bxs, scrs, lbls = gpu_nms(
                pred_boxes,
                pred_scores,
                num_class,
                max_boxes=30,
                score_thresh=0.5,
                nms_thresh=0.5,
            )

            restore_path = args.restore_path
            scheme = urlparse(restore_path).scheme
            # If there is a file scheme, it may be remote data hosted on s3/gcs.
            # In that case, cache the weights locally.
            if scheme:
                import fsspec
                from fsspec.implementations.cached import WholeFileCacheFileSystem

                kwargs = {"anon": True} if scheme == "s3" or scheme == "gcs" else {}
                path = fsspec.core.strip_protocol(restore_path)
                basename = os.path.basename(path)
                dirname = os.path.dirname(path)
                target = fsspec.filesystem(scheme, **kwargs)
                cache_dir = "/tmp/awbc/"
                fs = WholeFileCacheFileSystem(
                    fs=target, cache_storage=cache_dir, same_names=True,
                )
                # Get the object list, filtering out the directory itself.
                objects = [o for o in fs.ls(dirname) if o.rstrip("/") != dirname]
                # Trigger caching of the objects
                for o in objects:
                    logging.debug(f"Caching {o}")
                    fs.open(o)
                # Set the restore path to be the cached index
                restore_path = os.path.join(cache_dir, basename)

            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

            while (
                camera.isOpened() and not self.stop_thread or not self.stop_thread.get()
            ):

                self.current_frame_number += 1

                # Check for the need to generate the value of counters periodically
                if self.periodic_counter_interval != 0:
                    if self.current_frame_number % self.periodic_counter_interval == 0:
                        self.object_counter.export_counter_threading()

                elapsed = self.current_frame_number

                ret, img_ori = camera.read()

                print("Frame Number: " + str(elapsed))
                if img_ori is None:
                    print("\nEnd of Video")
                    break

                height_ori, width_ori = img_ori.shape[:2]

                if self.masked_image != []:
                    masked = cv2.bitwise_and(img_ori, self.masked_image)
                    img = cv2.resize(masked, tuple(args.new_size))
                    mask_inv = cv2.bitwise_not(self.masked_image)
                    img_ori = cv2.addWeighted(
                        img_ori,
                        1 - (self.output_video.opaque / 100),
                        mask_inv,
                        (self.output_video.opaque / 100),
                        0,
                    )
                else:
                    img = cv2.resize(img_ori, tuple(args.new_size))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.asarray(img, np.float32)
                img = img[np.newaxis, :] / 255.0

                boxes_, scores_, labels_ = sess.run(
                    [bxs, scrs, lbls], feed_dict={input_data: img}
                )

                boxes, boxes_drawing = self.convert_y3_boxes_to_boxes(
                    boxes_, scores_, labels_, img_ori, args.new_size, classes
                )

                postprocessed = img_ori

                self.current_frame = Frame(postprocessed, boxes)

                nodup_objects = self.current_frame.get_no_duplicate_objects()

                noins_objects = self.current_frame.remove_objects_inside_other_objects(
                    nodup_objects
                )

                detected_objects = self.create_contour_for_current_objects(
                    noins_objects
                )

                # first moving object
                if len(self.last_frame_moving_objects) == 0:
                    # for cont in contours:
                    for obj in noins_objects:
                        self.add_new_moving_object(obj)

                    if save_video:
                        video_writer.write(self.current_frame.postprocessed_frame)
                    if self.frame_listener:
                        self.update_frame_listener(
                            self.current_frame.postprocessed_frame,
                            self.current_frame_number,
                        )

                    if self.input_camera_type == "webcam" and not config.cli:
                        cv2.imshow("", self.current_frame.postprocessed_frame)

                # from the 2nd frame, calculate cost using predicted position and new
                # contour positions
                else:
                    # save all the positions to a matrix and calculate the cost
                    # initiate a matrix for calculating assianment by Hungarian
                    # algorithm. When no contour found in this frame then update kalman
                    # filter and skip

                    if len(detected_objects) == 0:
                        n = n + 1
                        self.update_skipped_frame(config.missing_threshold)

                        if save_video:
                            video_writer.write(self.current_frame.postprocessed_frame)
                        if self.frame_listener:
                            self.update_frame_listener(
                                self.current_frame.postprocessed_frame,
                                self.current_frame_number,
                            )

                        if self.input_camera_type == "webcam" and not config.cli:
                            temp_image = self.current_frame.postprocessed_frame
                            cv2.imshow("", temp_image)

                        continue

                    # matrix_h is the distance between all moving objects of previous
                    # frame and the current frame moving objects' centers
                    (
                        matrix_h,
                        cur_frame_available_moving_objects,
                    ) = self.calculate_cost_matrix_for_moving_objects(detected_objects)

                    logging.debug(f"Distance matrix: {str(matrix_h)}")
                    # when matrix is empty, skip this frame
                    if len(matrix_h) < 1:

                        n = n + 1
                        self.update_skipped_frame(config.missing_threshold)

                        if save_video:
                            video_writer.write(self.current_frame.postprocessed_frame)
                        if self.frame_listener:
                            self.update_frame_listener(
                                self.current_frame.postprocessed_frame,
                                self.current_frame_number,
                            )

                        if self.input_camera_type == "webcam" and not config.cli:
                            cv2.imshow("", self.current_frame.postprocessed_frame)

                        continue

                    self.predict_moving_objects_new_position(
                        matrix_h, cur_frame_available_moving_objects, detected_objects
                    )

                    self.print_data_report_on_frame()

                    if save_video:
                        video_writer.write(self.current_frame.postprocessed_frame)
                    if self.frame_listener:
                        self.update_frame_listener(
                            self.current_frame.postprocessed_frame,
                            self.current_frame_number,
                        )

                    if self.input_camera_type == "webcam" and not config.cli:
                        cv2.imshow("", self.current_frame.postprocessed_frame)

                if elapsed % 5 == 0:
                    logging.debug(
                        f"Processed frames per second: {(elapsed/timer()-start):3.3f}"
                    )
                if self.input_camera_type == "webcam" and not config.cli:
                    choice = cv2.waitKey(1)
                    if choice == 27:
                        break

        if save_video:
            video_writer.release()
        camera.release()
        if self.input_camera_type == "webcam" and not config.cli:
            cv2.destroyAllWindows()

        self.object_counter.export_counter_threading()

        # TODO: this should respect valid selected objects?
        final_count = (
            "Pedestrians: "
            + str(self.object_counter.COUNTER_p)
            + " Cyclists: "
            + str(self.object_counter.COUNTER_c)
        )
        print(final_count)

        self.object_counter.counter_thread.join()

    def remove_tracked_objects(self, thresh):

        for index, obj in enumerate(self.last_frame_moving_objects):
            obj.frames_since_seen += 1

            # if a moving object hasn't been updated for 10 frames then remove it
            if obj.frames_since_seen > thresh:
                del self.last_frame_moving_objects[index]

            # if the object is out of the scene then remove from current tracking right
            # away
            h, w = self.current_frame.postprocessed_frame.shape[:2]
            if obj.position[-1][0] < 0 or obj.position[-1][0] > w:
                del self.last_frame_moving_objects[index]

            elif obj.position[-1][1] < 0 or obj.position[-1][1] > h:
                del self.last_frame_moving_objects[index]

            elif self.masked_image != [] and not self.check_object_is_in_aoi(obj):
                del self.last_frame_moving_objects[index]

    def predict_moving_objects_new_position(
        self, cost_matrix, available_tracked_moving_objects, cur_detected_objects
    ):

        munkres = Munkres()

        indexes = munkres.compute(cost_matrix)

        total = 0
        for row, column in indexes:
            value = cost_matrix[row][column]
            total += value

        indexes_np = np.array(indexes)

        contour_index_list = indexes_np[:, 1].tolist()

        for detected_object_index, detected_object in enumerate(cur_detected_objects):

            if detected_object_index in indexes_np[:, 1]:

                index_m = contour_index_list.index(detected_object_index)

                tracked_obj_index = indexes_np[index_m, 0]

                threshold = config.ped_cost_threshold
                if (
                    available_tracked_moving_objects[
                        tracked_obj_index
                    ].last_detected_object.mess
                    == "bus"
                ):
                    threshold = config.bus_cost_threshold
                elif (
                    available_tracked_moving_objects[
                        tracked_obj_index
                    ].last_detected_object.mess
                    == "truck"
                ):
                    threshold = config.truck_cost_threshold

                if cost_matrix[tracked_obj_index][detected_object_index] > threshold:
                    print(
                        "\tObject ID "
                        + str(available_tracked_moving_objects[tracked_obj_index].id)
                        + " will be added as a new object because of the cost threshold"
                    )
                    self.add_new_moving_object(detected_object)
                    continue

                print(
                    "\tObject ID "
                    f"{available_tracked_moving_objects[tracked_obj_index].id}"
                    " has been assigned to object detected at "
                    f"{cur_detected_objects[detected_object_index].left:.0f} "
                    f"{cur_detected_objects[detected_object_index].right:.0f} "
                    f"{cur_detected_objects[detected_object_index].top:.0f} "
                    f"{cur_detected_objects[detected_object_index].bot:.0f}"
                )

                obj_m = available_tracked_moving_objects[tracked_obj_index]
                # get corresponding contour position, update kalman filter
                position_new = cur_detected_objects[detected_object_index].center
                obj_m.last_detected_object = cur_detected_objects[detected_object_index]
                obj_m.kalman_update(position_new)
                obj_m.counted += 1
                self.add_new_moving_object_to_counter(
                    obj_m, position_new, self.current_frame.postprocessed_frame
                )

                self.draw_boxes_on_frame(obj_m)

            else:
                position_new = cur_detected_objects[detected_object_index]
                self.add_new_moving_object(position_new)

        # these are tracks missed either because they disappeared
        # or because they are temporarily invisable
        for index, obj in enumerate(available_tracked_moving_objects):
            if index not in indexes_np[:, 0]:
                # not update in this frame, increase frames_since_seen
                obj.frames_since_seen += 1
                # but we update KF with predicted location
                obj.kalman_update_missing(obj.predicted_position[-1])
                print("\tObject " + str(obj.id) + " has disappeared from this frame.")

        # remove movingObj not updated for more than threasholds numbers of frames
        for index, obj in enumerate(self.last_frame_moving_objects):

            h, w = self.current_frame.postprocessed_frame.shape[:2]

            self.check_object_for_deletion(obj, index)

    def check_object_for_deletion(self, obj, index):

        if obj.frames_since_seen > config.missing_threshold:
            if (
                obj.position[-1][0] < self.BOUNDRY
                or obj.position[-1][0] > self.video_width - self.BOUNDRY
                or obj.position[-1][1] < self.BOUNDRY
                or obj.position[-1][1] > self.video_height - self.BOUNDRY
            ):
                print(f"\tDeleting {obj.id} as it has left the frame")
                del self.last_frame_moving_objects[index]
            elif obj.frames_since_seen > config.missing_threshold:
                # TODO: this if check seems like a bug: redundant
                print(f"\tDeleting {obj.id} as it has disappeared from the frame")
                del self.last_frame_moving_objects[index]

    def convert_y3_boxes_to_boxes(
        self, boxes_, scores_, labels_, original_image, new_size, object_class
    ):

        height_ori, width_ori = original_image.shape[:2]

        boxes_[:, 0] *= width_ori / float(new_size[0])
        boxes_[:, 2] *= width_ori / float(new_size[0])
        boxes_[:, 1] *= height_ori / float(new_size[1])
        boxes_[:, 3] *= height_ori / float(new_size[1])

        # (left, right, top, bot, mess, max_indx, confidence)

        boxes_counting = []
        boxes_drawing = []
        for i in range(0, len(boxes_)):
            mess = object_class[labels_[i]]
            if mess in self.object_classes:
                cx = boxes_[i][0] + int((boxes_[i][2] - boxes_[i][0]) / 2)
                cy = boxes_[i][1] + int((boxes_[i][3] - boxes_[i][1]) / 2)

                # Specifying the area of interest
                if (0 < cx < width_ori) and (0 < cy < height_ori):
                    if labels_[i] < 8:
                        boxes_counting.append(
                            (
                                boxes_[i][0],
                                boxes_[i][2],
                                boxes_[i][1],
                                boxes_[i][3],
                                object_class[labels_[i]],
                                0,
                                scores_[i],
                            )
                        )
                        boxes_drawing.append(
                            (
                                boxes_[i][0],
                                boxes_[i][2],
                                boxes_[i][1],
                                boxes_[i][3],
                                object_class[labels_[i]],
                                0,
                                scores_[i],
                            )
                        )

        return boxes_counting, boxes_drawing

    def draw_boxes_on_frame(self, obj):
        x0 = obj.last_detected_object.left
        y0 = obj.last_detected_object.top
        x1 = obj.last_detected_object.right
        y1 = obj.last_detected_object.bot
        mess = obj.last_detected_object.mess

        img_ori = self.current_frame.postprocessed_frame

        if mess == "person":
            if obj.id in self.object_counter.Pedestrians:
                plot_one_box(
                    img_ori, [x0, y0, x1, y1], label="", color=self.color_table[mess]
                )
            elif obj.id in self.object_counter.Cyclists:
                plot_one_box(
                    img_ori,
                    [x0, y0, x1, y1],
                    label="Cyclist",
                    color=self.color_table["cyclist"],
                )
        elif mess == "bicycle":
            # TODO: what is the point of this branch?
            print("Bicycle detected...")
        elif obj.id in self.object_counter.Cars:
            plot_one_box(
                img_ori, [x0, y0, x1, y1], label="Car", color=self.color_table["car"]
            )
        elif obj.id in self.object_counter.Trucks:
            plot_one_box(
                img_ori,
                [x0, y0, x1, y1],
                label="Truck",
                color=self.color_table["truck"],
            )

    def update_frame_listener(self, frame, frame_number):
        self.frame_listener(frame, frame_number)

    def check_object_is_in_aoi(self, obj):

        return True
