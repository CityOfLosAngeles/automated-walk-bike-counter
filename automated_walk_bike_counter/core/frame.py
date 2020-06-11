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

import numpy as np

from .bounding_box.biker import Biker
from .bounding_box.bus import Bus
from .bounding_box.car import Car
from .bounding_box.motorbiker import MotorBiker
from .bounding_box.pedestrian import Pedestrian
from .bounding_box.truck import Truck


class Frame:
    # duplicated classification threshold > 0.30 then means 2 classification for same
    # object
    DUPLICATE_THRESHOLD = 0.30
    DUPLICATE_CAR_THRESHOLD = 0.90
    DUPLICATE_TRUCK_THRESHOLD = 0.90
    CAR_TRUCK_DUPLICATE_THRESHOLD = 0.80
    BIK_AREA_THRESHOLD = 700

    def __init__(self, postprocessed, boxes):
        self.pedestrians = []
        self.bikers = []
        self.motorbikers = []
        self.cars = []
        self.buses = []
        self.trucks = []
        self.postprocessed_frame = postprocessed
        self.boxes = boxes
        self.create_detected_object()

    def create_detected_object(self):
        for box in self.boxes:
            (left, right, top, bot, mess, max_indx, confidence) = box

            if mess == "person":
                self.pedestrians.append(Pedestrian(box))
            elif mess == "bicycle":
                self.bikers.append(Biker(box))
            elif mess == "motorbike":
                self.motorbikers.append(MotorBiker(box))
            elif mess == "car":
                self.cars.append(Car(box))
            elif mess == "bus":
                self.buses.append(Bus(box))
            elif mess == "truck":
                self.trucks.append(Truck(box))

    def find_duplicated_objects(self):
        def overlap_area(boxes):
            if len(boxes) == 0:
                return 0

            xx1 = max(boxes[0, 0], boxes[1, 0])
            yy1 = max(boxes[0, 1], boxes[1, 1])
            xx2 = min(boxes[0, 2], boxes[1, 2])
            yy2 = min(boxes[0, 3], boxes[1, 3])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # box1 area
            area1 = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
            # box2 area
            area2 = (boxes[1, 2] - boxes[1, 0]) * (boxes[1, 3] - boxes[1, 1])

            area = min(area1, area2)
            overlap = (w * h) / area

            return overlap

        ped_boxes_dup_dict = {}
        # compare bike/motorbike and pedestrian, find duplicates
        for bik in self.bikers:
            for ped in self.pedestrians:
                boxes_2compare = np.array(
                    [
                        [ped.left, ped.top, ped.right, ped.bot],
                        [bik.left, bik.top, bik.right, bik.bot],
                    ]
                )
                o_rate = overlap_area(boxes_2compare)
                print("overlap: ", o_rate)
                if o_rate > self.DUPLICATE_THRESHOLD:
                    ped_boxes_dup_dict[ped] = 1
                    print(
                        "exclude for duplicates:", ped.left, ped.right, ped.top, ped.bot
                    )

        for ped1 in self.pedestrians:
            for ped2 in self.pedestrians:
                if ped1 is not ped2:
                    boxes_2compare = np.array(
                        [
                            [ped1.left, ped1.top, ped1.right, ped1.bot],
                            [ped2.left, ped2.top, ped2.right, ped2.bot],
                        ]
                    )
                    o_rate = overlap_area(boxes_2compare)
                    if o_rate > self.DUPLICATE_THRESHOLD:
                        ped_boxes_dup_dict[ped2] = 1

        for mot in self.motorbikers:
            for ped in self.pedestrians:
                boxes_2compare = np.array(
                    [
                        [ped.left, ped.top, ped.right, ped.bot],
                        [mot.left, mot.top, mot.right, mot.bot],
                    ]
                )
                o_rate = overlap_area(boxes_2compare)
                print("overlap: ", o_rate)
                if o_rate > self.DUPLICATE_THRESHOLD:
                    ped_boxes_dup_dict[ped] = 1
                    print(
                        "exclude for duplicates:", ped.left, ped.right, ped.top, ped.bot
                    )

        for car1 in self.cars:
            for car2 in self.cars:
                if car1 is not car2:
                    boxes_2compare = np.array(
                        [
                            [car1.left, car1.top, car1.right, car1.bot],
                            [car2.left, car2.top, car2.right, car2.bot],
                        ]
                    )
                    o_rate = overlap_area(boxes_2compare)
                    print("overlap: ", o_rate)
                    if o_rate > self.DUPLICATE_CAR_THRESHOLD:
                        ped_boxes_dup_dict[car2] = 1
                        print(
                            "car exclude for car duplicates:",
                            car2.left,
                            car2.right,
                            car2.top,
                            car2.bot,
                        )

        for truck1 in self.trucks:
            for truck2 in self.trucks:
                if truck1 is not truck2:
                    boxes_2compare = np.array(
                        [
                            [truck1.left, truck1.top, truck1.right, truck1.bot],
                            [truck2.left, truck2.top, truck2.right, truck2.bot],
                        ]
                    )
                    o_rate = overlap_area(boxes_2compare)
                    print("overlap: ", o_rate)
                    if o_rate > self.DUPLICATE_TRUCK_THRESHOLD:
                        ped_boxes_dup_dict[truck2] = 1
                        print(
                            "truck exclude for truck duplicates:",
                            truck2.left,
                            truck2.right,
                            truck2.top,
                            truck2.bot,
                        )

        for car in self.cars:
            for truck in self.trucks:

                boxes_2compare = np.array(
                    [
                        [car.left, car.top, car.right, car.bot],
                        [truck.left, truck.top, truck.right, truck.bot],
                    ]
                )
                o_rate = overlap_area(boxes_2compare)
                print("overlap: ", o_rate)
                if o_rate > self.CAR_TRUCK_DUPLICATE_THRESHOLD:
                    ped_boxes_dup_dict[truck] = 1
                    print(
                        "truck exclude for duplicates with car:",
                        truck.left,
                        truck.right,
                        truck.top,
                        truck.bot,
                    )

        for car in self.cars:
            for ped in self.pedestrians:
                boxes_2compare = np.array(
                    [
                        [ped.left, ped.top, ped.right, ped.bot],
                        [car.left, car.top, car.right, car.bot],
                    ]
                )
                o_rate = overlap_area(boxes_2compare)
                print("overlap: ", o_rate)
                if o_rate > self.DUPLICATE_THRESHOLD:
                    ped_boxes_dup_dict[ped] = 1
                    print(
                        "Car excluded for duplication with pedestrian :",
                        ped.left,
                        ped.right,
                        ped.top,
                        ped.bot,
                    )

        return ped_boxes_dup_dict

    def get_no_duplicate_objects(self):

        no_duplicate_objects = []
        ped_boxes_dup_dict = self.find_duplicated_objects()

        for car in self.cars:
            no_duplicate_objects.append(car)

        # add pedestrian only into nodup_boxes
        for ped in self.pedestrians:
            if ped in ped_boxes_dup_dict:
                continue
            else:
                no_duplicate_objects.append(ped)

        # add bike into nodup_boxes
        for bik in self.bikers:
            # add filter to exclude bicycle bbox too small, could be a false detection
            # of a handbag
            if (
                (bik.right - bik.left) * (bik.bot - bik.top) < self.BIK_AREA_THRESHOLD
                and (bik.bot - bik.top) < (bik.right - bik.left) * 1.5
            ):
                continue
            else:
                no_duplicate_objects.append(bik)

        # add motorbikers 7/27, since we need to do better job for excluding motorbikers
        for mot in self.motorbikers:
            no_duplicate_objects.append(mot)

        for truck in self.trucks:
            if truck in ped_boxes_dup_dict:
                continue
            else:
                no_duplicate_objects.append(truck)

        return no_duplicate_objects

    def remove_objects_inside_other_objects(self, list_of_objects):
        insider_dict = {}
        no_inside_objects = []

        margin = 10
        # remove any box that inside of other boxes
        for nbox in list_of_objects:
            for mbox in list_of_objects:
                if nbox.box == mbox.box:
                    continue
                # check since inside conditions is valid for ped and bicycles and not
                # for cars
                if not (
                    (nbox.box[4] == "car" and mbox.box[4] == "person")
                    or (nbox.box[4] == "person" and mbox.box[4] == "car")
                ):
                    if (
                        mbox.left + margin >= nbox.left
                        and mbox.right - margin <= nbox.right
                        and mbox.top + margin >= nbox.top
                        and mbox.bot - margin <= nbox.bot
                    ):
                        insider_dict[mbox] = 1  # ---- item caused different counter
                        print(
                            mbox.left,
                            mbox.right,
                            mbox.top,
                            mbox.bot,
                            " inside of ",
                            nbox.left,
                            nbox.right,
                            nbox.top,
                            nbox.bot,
                        )

        for obj in list_of_objects:
            if obj in insider_dict:
                continue
            else:
                no_inside_objects.append(obj)

        return no_inside_objects
