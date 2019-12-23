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

import math
import os
import re
import sys
from time import time as timer
from urllib.parse import urlparse

import cv2
import numpy as np
import s3fs
import tensorflow as tf
from munkres import Munkres

from ...utils.misc_utils import parse_anchors, read_class_names
from ...utils.nms_utils import gpu_nms
from ...utils.plot_utils import plot_one_box
from ..configuration import config
from ..frame import Frame
from ..model import yolov3
from ..movingobject import MovingObject
from ..tracking.counter import Object_Counter


class ObjectTracker:

    # PED_AREA_THRESHOLD = 600
    # PED_COST_THREASHOLD = 80
    # BUS_COST_THREASHOLD = 110
    # TRUCK_COST_THREASHOLD = 110
    # MISSING_THREASHOLD = 90
    # MISSING_THREASHOLD_MAX = 300

    BOUNDRY = 30

    COUNT_THRESHOLD = 10
    COUNT_THRESHOLD_BIKE = 5
    COUNT_THRESHOLD_MOTOR = 3

    def __init__(self, mask_image):
        self.lastFrameMovingObjects = []
        self.masked_image = mask_image
        self.moving_object_id_number = 0
        self.object_counter = Object_Counter()
        self.current_frame = None
        self.currentFrameNumber = 0  # elapsed frames
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

    def printDataReportOnFrame(self):
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
            counter = "  Fr:" + str(self.currentFrameNumber)
            cv2.putText(
                self.current_frame.postprocessed_frame,
                counter,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    def addNewMovingObjectToCounter(self, obj, position_new, postprocessed):
        self.object_counter.addNewMovingObjectForCounting(
            obj, position_new, postprocessed
        )

    def addNewMovingObject(self, current_detected_object):
        self.moving_object_id_number += 1
        new_mObject = MovingObject(
            self.moving_object_id_number, current_detected_object.center
        )
        new_mObject.last_detected_object = current_detected_object
        new_mObject.add_position([current_detected_object.center])
        new_mObject.init_kalman_filter()
        filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(
            new_mObject.position
        )
        new_mObject.set_next_mean(filtered_state_means[-1])
        new_mObject.set_next_covariance(filtered_state_covariances[-1])
        new_mObject.counted += 1

        print(
            "New moving object added with the id of "
            + str(self.moving_object_id_number)
            + " detected as "
            + current_detected_object.mess
            + " in position : "
            + str(current_detected_object.left)
            + " "
            + str(current_detected_object.right)
            + " "
            + str(current_detected_object.top)
            + " "
            + str(current_detected_object.bot)
        )
        # add to current_tracks
        self.lastFrameMovingObjects.append(new_mObject)

    def createContourForCurrentObjects(self, detected_objects):

        detectedObjectsWithValidContours = []

        for obj in detected_objects:
            (left, right, top, bot, mess, max_indx, confidence) = obj.box
            # 04/03/2018 exclude pedestrian inside of car by bbox size and shape

            # commented since time of testing car countters whihch was caused problem
            # for ped counter
            # if (
            #     mess=='person' and
            #     (
            #         ((right-left)*(bot-top) < self.PED_AREA_THRESHOLD) or
            #         (bot-top) < (right-left) * 1.5)
            # ):
            # 	continue

            detectedObjectsWithValidContours.append(obj)

        return detectedObjectsWithValidContours

    def calculateCostMatrixForMovingObjects(self, currentDetectedObjects):
        def get_costs(curObject, cur_detected_objects):
            distances = []
            for obj in cur_detected_objects:

                dis = math.floor(
                    math.sqrt(
                        (obj.center_x - curObject.predicted_position[-1][0]) ** 2
                        + (obj.center_y - curObject.predicted_position[-1][1]) ** 2
                    )
                )
                dis += abs(
                    self.object_costs[curObject.last_detected_object.mess]
                    - self.object_costs[obj.mess]
                )
                distances.append(dis)
                print(
                    "Distance between obj known "
                    + str(obj.mess)
                    + " at position "
                    + str(obj.left)
                    + " "
                    + str(obj.right)
                    + " "
                    + str(obj.top)
                    + " "
                    + str(obj.bot)
                    + " with object with id "
                    + str(curObject.id)
                    + " known as "
                    + curObject.last_detected_object.mess
                    + " is "
                    + str(dis)
                )
            return distances

        def get_costs_extended(last_frame_detected, cur_detected_objects):
            distances = []
            coefficient = 1
            for obj in cur_detected_objects:
                distances.append(
                    math.floor(
                        math.sqrt(
                            (
                                obj.center_x
                                - last_frame_detected.predicted_position[-1][0]
                            )
                            ** 2
                            + (
                                obj.center_y
                                - last_frame_detected.predicted_position[-1][1]
                            )
                            ** 2
                        )
                    )
                    + coefficient
                    * math.floor(
                        math.sqrt(
                            (
                                obj.getWidth()
                                - last_frame_detected.last_detected_object.getWidth()
                            )
                            ** 2
                            + (
                                obj.getHeigth()
                                - last_frame_detected.last_detected_object.getHeigth()
                            )
                            ** 2
                        )
                    )
                )
            return distances

        lastFrameMovingObjectsCostMatrix = []
        validMovingObjects = []
        for index, obj in enumerate(self.lastFrameMovingObjects):
            # calculate costs for each tracked movingObjects using their predicted
            # position
            costs = get_costs(obj, currentDetectedObjects)

            # if moving object to all contours distances are too large, then not to
            # consider it at all
            threshold = config.PED_COST_THRESHOLD
            if obj.last_detected_object.mess == "bus":
                threshold = config.BUS_COST_THRESHOLD
            elif obj.last_detected_object.mess == "truck":
                threshold = config.TRUCK_COST_THRESHOLD

            if all(c > threshold for c in costs):
                print(
                    "object id "
                    + str(obj.id)
                    + " cost with all other detected objects is more than threshold "
                    + "and is counted as missing in kalman"
                )
                # update it with KF predicted position
                obj.kalman_update_missing(obj.predicted_position[-1])
                # skip this moving object
                continue

            lastFrameMovingObjectsCostMatrix.append(costs)
            # only valid moving objects are added to available_objecs
            validMovingObjects.append(obj)

        return lastFrameMovingObjectsCostMatrix, validMovingObjects

    def update_skipped_frame(
        self, thresh,
    ):

        self.printDataReportOnFrame()

        self.removeTrackedObjects(thresh)
        print("update skipped frame")

    def trackObjects(self, args):

        anchors = parse_anchors(args.anchor_path)
        classes = read_class_names(args.class_name_path)
        num_class = len(classes)

        self.image_processing_size = args.new_size

        file = self.video_filename
        SaveVideo = args.save_video
        print("save : " + str(SaveVideo))

        print("video : ", file)

        # check if the video is reading from a file or from the webcam
        if file == "camera":
            file = 0
            vfname = "camera"
        else:
            # get the video name to process
            m = re.match(r"([^\.]*)(\..*)", file)
            vfname = m.group(1)

            assert os.path.isfile(file), "file {} does not exist".format(file)

        camera = cv2.VideoCapture(file)

        self.video_width = int(camera.get(3))
        self.video_height = int(camera.get(4))

        # if file == 0:
        # 	tfnetObject.say('Press [ESC] to quit demo')

        assert camera.isOpened(), "Cannot capture source"

        if file == 0:  # camera window
            cv2.namedWindow("", 0)
            # _, frame = camera.read()
            # height, width, _ = frame.shape
            cv2.resizeWindow("", self.video_width, self.video_height)

        if SaveVideo:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            outfile = vfname + "_result.mp4"
            print(outfile)
            if file == 0:  # camera window
                # fps = 1 / tfnetObject._get_fps(frame)
                # TODO What is FPS here?
                if fps < 1:  # noqa: F821
                    fps = 1
            else:
                fps = round(camera.get(cv2.CAP_PROP_FPS))

            videoWriter = cv2.VideoWriter(
                outfile, fourcc, fps, (self.video_width, self.video_height)
            )

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
            yolo_model = yolov3(num_class, anchors)
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

            # Set the AWS region for the Tensorflow S3 adapter.
            # TODO: figure out a more robust way of handling this, so
            # that we can have multiple regions or different storage backends.
            restore_path = args.restore_path
            os.environ["AWS_REGION"] = config.AWS_REGION
            if urlparse(restore_path).scheme == "s3" and sys.platform == "win32":
                print("Windows detected -- caching s3 artifacts")
                fs = s3fs.S3FileSystem(anon=True)
                dirname = os.path.dirname(restore_path)
                objects = fs.ls(dirname)[1:]  # The first entry is the directory itself
                cache = os.path.join(os.path.expanduser("~"), ".awbc")
                if not os.path.exists(cache):
                    os.mkdir(cache)
                for obj in objects:
                    print(f"Copying {obj}")
                    out = os.path.join(cache, os.path.basename(obj))
                    if not os.path.exists(out):
                        with fs.open(obj, "rb") as ifile:
                            with open(out, "wb") as ofile:
                                ofile.write(ifile.read())
                restore_path = os.path.join(cache, os.path.basename(restore_path))
                print("Restoring from cache: ", restore_path)

            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

            while camera.isOpened():

                self.currentFrameNumber += 1

                elapsed = self.currentFrameNumber

                ret, img_ori = camera.read()

                print("Frame No. : " + str(elapsed))
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

                boxes, boxes_drawing = self.convertY3BoxesToBoxes(
                    boxes_, scores_, labels_, img_ori, args.new_size, classes
                )

                postprocessed = img_ori

                self.current_frame = Frame(postprocessed, boxes)

                nodup_objects = self.current_frame.getNoDuplicateObjects()

                noins_objects = self.current_frame.removeObjectsInsideOtherObjects(
                    nodup_objects
                )

                detected_objects = self.createContourForCurrentObjects(noins_objects)

                # first moving object
                if len(self.lastFrameMovingObjects) == 0:
                    # for cont in contours:
                    for obj in noins_objects:
                        self.addNewMovingObject(obj)

                    if SaveVideo:
                        videoWriter.write(self.current_frame.postprocessed_frame)
                        self.update_frame_listener(
                            self.current_frame.postprocessed_frame,
                            self.currentFrameNumber,
                        )
                    if file == 0:  # camera window
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
                        self.update_skipped_frame(config.MISSING_THRESHOLD)

                        if SaveVideo:
                            videoWriter.write(self.current_frame.postprocessed_frame)
                            self.update_frame_listener(
                                self.current_frame.postprocessed_frame,
                                self.currentFrameNumber,
                            )
                        if file == 0:  # camera window
                            cv2.imshow("", self.current_frame.postprocessed_frame)

                        continue

                    # matrix_h is the distance between all moving objects of previous
                    # frame and the current frame moving objects' centers
                    (
                        matrix_h,
                        cur_frame_available_moving_objects,
                    ) = self.calculateCostMatrixForMovingObjects(detected_objects)

                    print(str(matrix_h))
                    # when matrix is empty, skip this frame
                    if len(matrix_h) < 1:

                        n = n + 1
                        self.update_skipped_frame(config.MISSING_THRESHOLD)

                        if SaveVideo:
                            videoWriter.write(self.current_frame.postprocessed_frame)
                            self.update_frame_listener(
                                self.current_frame.postprocessed_frame,
                                self.currentFrameNumber,
                            )
                        if file == 0:  # camera window
                            cv2.imshow("", self.current_frame.postprocessed_frame)

                        continue

                    self.predictMovingObjectsNewPosition(
                        matrix_h, cur_frame_available_moving_objects, detected_objects
                    )

                    self.printDataReportOnFrame()

                    if SaveVideo:
                        videoWriter.write(self.current_frame.postprocessed_frame)
                        self.update_frame_listener(
                            self.current_frame.postprocessed_frame,
                            self.currentFrameNumber,
                        )

                    if file == 0:  # camera window
                        cv2.imshow("", self.current_frame.postprocessed_frame)

                if elapsed % 5 == 0:
                    sys.stdout.write("\r")
                    sys.stdout.write("{0:3.3f} FPS".format(elapsed / (timer() - start)))
                    sys.stdout.flush()
                if file == 0:  # camera window
                    choice = cv2.waitKey(1)
                    if choice == 27:
                        break

        if SaveVideo:
            videoWriter.release()
        camera.release()
        if file == 0:  # camera window
            cv2.destroyAllWindows()

        count = (
            "Pedestrians: "
            + str(self.object_counter.COUNTER_p)
            + " Cyclists: "
            + str(self.object_counter.COUNTER_c)
        )
        print(count)

    def removeTrackedObjects(self, thresh):
        # MISSING_THREASHOLD = 90

        for index, obj in enumerate(self.lastFrameMovingObjects):
            obj.frames_since_seen += 1

            # if a moving object hasn't been updated for 10 frames then remove it
            if obj.frames_since_seen > thresh:
                del self.lastFrameMovingObjects[index]

            # if the object is out of the scene then remove from current tracking right
            # away
            h, w = self.current_frame.postprocessed_frame.shape[:2]
            if obj.position[-1][0] < 0 or obj.position[-1][0] > w:
                del self.lastFrameMovingObjects[index]

            elif obj.position[-1][1] < 0 or obj.position[-1][1] > h:
                del self.lastFrameMovingObjects[index]

            elif self.masked_image != [] and not self.check_object_is_in_aoi(obj):
                print(
                    "Delete tracking object ",
                    obj.position[-1][0],
                    obj.position[-1][1],
                    "out of the area of interest!}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}",
                )
                del self.lastFrameMovingObjects[index]

    def predictMovingObjectsNewPosition(
        self, costMatrix, availableTrackedMovingObjects, cur_detected_objects
    ):

        munkres = Munkres()

        indexes = munkres.compute(costMatrix)

        print("indexes = " + str(indexes))

        total = 0
        for row, column in indexes:
            value = costMatrix[row][column]
            total += value

        indexes_np = np.array(indexes)

        print("index_np : " + str(indexes_np))

        contour_index_list = indexes_np[:, 1].tolist()

        print("contour_index_list : " + str(contour_index_list))

        for detected_object_index, detected_object in enumerate(cur_detected_objects):

            if detected_object_index in indexes_np[:, 1]:

                index_m = contour_index_list.index(detected_object_index)

                tracked_obj_index = indexes_np[index_m, 0]

                threshold = config.PED_COST_THRESHOLD
                if (
                    availableTrackedMovingObjects[
                        tracked_obj_index
                    ].last_detected_object.mess
                    == "bus"
                ):
                    threshold = config.BUS_COST_THRESHOLD
                elif (
                    availableTrackedMovingObjects[
                        tracked_obj_index
                    ].last_detected_object.mess
                    == "truck"
                ):
                    threshold = config.TRUCK_COST_THRESHOLD

                if costMatrix[tracked_obj_index][detected_object_index] > threshold:
                    print(
                        "Object id "
                        + str(availableTrackedMovingObjects[tracked_obj_index].id)
                        + " is going to add as a new object because of cost threshold"
                    )
                    self.addNewMovingObject(detected_object)
                    continue

                print(
                    "object "
                    + str(availableTrackedMovingObjects[tracked_obj_index].id)
                    + " has been assigned to object detected at "
                    + " position : "
                    + str(cur_detected_objects[detected_object_index].left)
                    + " "
                    + str(cur_detected_objects[detected_object_index].right)
                    + " "
                    + str(cur_detected_objects[detected_object_index].top)
                    + " "
                    + str(cur_detected_objects[detected_object_index].bot)
                )

                obj_m = availableTrackedMovingObjects[tracked_obj_index]
                # get corresponding contour position, update kalman filter
                position_new = cur_detected_objects[detected_object_index].center
                obj_m.last_detected_object = cur_detected_objects[detected_object_index]
                obj_m.kalman_update(position_new)
                obj_m.counted += 1
                print("counted " + str(obj_m.id) + " " + str(obj_m.counted))
                self.addNewMovingObjectToCounter(
                    obj_m, position_new, self.current_frame.postprocessed_frame
                )

                self.drawBoxesOnFrame(obj_m)

            else:
                position_new = cur_detected_objects[detected_object_index]
                self.addNewMovingObject(position_new)

        # these are tracks missed either because they disappeared
        # or because they are temporarily invisable
        for index, obj in enumerate(availableTrackedMovingObjects):
            if index not in indexes_np[:, 0]:
                # not update in this frame, increase frames_since_seen
                obj.frames_since_seen += 1
                # but we update KF with predicted location
                obj.kalman_update_missing(obj.predicted_position[-1])
                print(
                    "object id " + str(obj.id) + " has been disappeared in this frame"
                )

        # remove movingObj not updated for more than threasholds numbers of frames
        for index, obj in enumerate(self.lastFrameMovingObjects):

            h, w = self.current_frame.postprocessed_frame.shape[:2]

            self.checkObjectForDeletion(obj, index)

    def checkObjectForDeletion(self, obj, index):

        if obj.frames_since_seen > config.MISSING_THRESHOLD:
            if (
                obj.position[-1][0] < self.BOUNDRY
                or obj.position[-1][0] > self.video_width - self.BOUNDRY
                or obj.position[-1][1] < self.BOUNDRY
                or obj.position[-1][1] > self.video_height - self.BOUNDRY
            ):
                print("Delete tracking", obj.position[-1][0], obj.position[-1][1])
                del self.lastFrameMovingObjects[index]
            elif obj.frames_since_seen > config.MISSING_THRESHOLD:
                print(
                    "object id: "
                    + str(obj.id)
                    + " frames_since_last_seen : "
                    + str(obj.frames_since_seen)
                )
                print(
                    obj.last_detected_object.box[4]
                    + " Delete tracking over max missing threshold"
                )
                del self.lastFrameMovingObjects[index]

    def convertY3BoxesToBoxes(
        self, boxes_, scores_, labels_, originalImage, new_size, object_class
    ):

        height_ori, width_ori = originalImage.shape[:2]

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
                # cx = self.left + int((self.right-self.left)/2)
                # cy = self.top + int((self.bot-self.top)/2)
                cx = boxes_[i][0] + int((boxes_[i][2] - boxes_[i][0]) / 2)
                cy = boxes_[i][1] + int((boxes_[i][3] - boxes_[i][1]) / 2)

                # Specifying the area of interest
                # if ((cx>0 and cx<width_ori) and (cy>160 and cy<height_ori)):
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

    def drawBoxesOnFrame(self, obj):
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
            print("Bicycle detected....")
        else:
            plot_one_box(
                img_ori, [x0, y0, x1, y1], label=mess, color=self.color_table[mess]
            )

    def update_frame_listener(self, frame, frame_number):
        self.frame_listener(frame, frame_number)

    def check_object_is_in_aoi(self, obj):

        r, g, b = self.masked_image[
            int(
                obj.position[-1][0] * (self.image_processing_size[0] / self.video_width)
            ),
            int(
                obj.position[-1][1]
                * (self.image_processing_size[1] / self.video_height)
            ),
        ]

        if r == 0 and g == 0 and b == 0:
            # The object is not in the area of interest
            return False
        else:
            return True
