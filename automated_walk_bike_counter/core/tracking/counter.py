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

import csv
import datetime
import os
import threading

from ..configuration import config


class ObjectCounter:
    Motorbikes = {}
    Duplicates = {}

    def __init__(self):
        self.COUNTER_m = 0
        self.COUNTER_p = 0
        self.COUNTER_c = 0
        self.COUNTER_o = 0
        self.COUNTER_car = 0
        self.COUNTER_bus = 0
        self.COUNTER_truck = 0

        self.Cars = {}
        self.Buses = {}
        self.Pedestrians = {}
        self.Cyclists = {}
        self.Trucks = {}

        self.output_counter_file_name = "counter"
        self.last_exported_ped_counter = 0
        self.last_exported_cyclist_counter = 0
        self.last_exported_car_counter = 0
        self.last_exported_bus_counter = 0
        self.last_exported_truck_counter = 0
        self.export_counter = 0
        self.counter_thread = None
        self.valid_selected_objects = []

    def add_new_moving_object_for_counting(self, obj, position_new, postprocessed):
        cur_detected_object = obj.last_detected_object
        cont_m = cur_detected_object.mess

        print("Counting for object id:" + str(obj.id) + " as " + cont_m)

        # for duplicated detection for bikers, when biker and motorbikers get detected
        # as pedestrian first
        if cont_m == "person" or cont_m == "bicycle" or cont_m == "motorbike":

            if obj.id in self.Pedestrians.keys():

                if cont_m == "bicycle" and obj.counted_biker >= 2:
                    # this is probably a biker not a pedestrian
                    self.COUNTER_p -= 1
                    self.COUNTER_c += 1
                    self.Cyclists[obj.id] = self.COUNTER_c
                    self.Pedestrians.pop(obj.id)
                    print(
                        "Person "
                        + str(obj.id)
                        + " counted for equal or more than 3 times for bicycle and "
                        + "detected as cyclist"
                    )

                elif cont_m == "bicycle" and obj.counted_biker < 2:
                    # increase counter
                    obj.counted_biker += 1
                    print(
                        "Person "
                        + str(obj.id)
                        + " counted for bicycle for "
                        + str(obj.counted_biker)
                        + " times."
                    )

                if cont_m == "motorbike" and obj.counted_moter >= 2:
                    # this is probably a moterbiker
                    self.COUNTER_p -= 1
                    self.COUNTER_o += 1
                    self.Motorbikes[obj.id] = self.COUNTER_o
                    self.Pedestrians.pop(obj.id)

                elif cont_m == "motorbike" and obj.counted_moter < 2:
                    obj.counted_moter += 1

            if (
                obj.id not in self.Pedestrians.keys()
                and obj.id not in self.Cyclists.keys()
                and obj.id not in self.Motorbikes.keys()
            ):

                if cont_m == "person" and obj.counted >= config.count_threshold:
                    print("counted person " + str(obj.id) + " " + str(obj.counted))
                    (position_x, position_y) = obj.position[-1]
                    self.COUNTER_p += 1
                    self.Pedestrians[obj.id] = self.COUNTER_p
                    obj.pedestrian_id = 1
                    # mark the moving object with the id
                elif cont_m == "bicycle" and obj.counted >= config.count_threshold_bike:
                    # ever detected as pedestrian, added 4/18 for prevent detecting
                    # bicycle without rider
                    if obj.pedestrian_id == 1:
                        print("counted bicycle " + str(obj.id) + " " + str(obj.counted))

                        self.COUNTER_c += 1
                        self.Cyclists[obj.id] = self.COUNTER_c
                        # mark the moving object with the id
                # added on 7/23
                elif (
                    cont_m == "motorbike"
                    and obj.counted >= config.count_threshold_motor
                ):
                    print("counted motorbike " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_o += 1
                    self.Motorbikes[obj.id] = self.COUNTER_o
        else:

            if (
                (obj.id not in self.Cars.keys())
                and (obj.id not in self.Buses.keys())
                and (obj.id not in self.Trucks.keys())
            ):
                if cont_m == "car" and obj.counted >= config.count_threshold_car:
                    print("counted car " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_car += 1
                    self.Cars[obj.id] = self.COUNTER_car
                    print(">>" + str(obj.id) + " detected as new car")

                elif cont_m == "bus" and obj.counted >= config.count_threshold_bus:
                    print("counted bus " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_bus += 1
                    self.Buses[obj.id] = self.COUNTER_bus
                    print(">>" + str(obj.id) + " detected as new bus")

                elif cont_m == "truck" and obj.counted >= config.count_threshold_truck:
                    print("counted truck " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_truck += 1
                    self.Trucks[obj.id] = self.COUNTER_truck
                    print(">>" + str(obj.id) + " detected as new truck")
            else:

                if obj.id in self.Trucks.keys():
                    if cont_m == "bus" and obj.counted_bus >= 3:
                        # this is probably a bus not a truck
                        self.COUNTER_truck -= 1
                        self.COUNTER_bus += 1
                        self.Buses[obj.id] = self.COUNTER_bus
                        self.Trucks.pop(obj.id)
                        print(
                            ">>" + str(obj.id) + " detected as a truck converted to bus"
                        )

                    elif cont_m == "car" and obj.counted_car >= 3:
                        # this is probably a bus not a truck
                        self.COUNTER_truck -= 1
                        self.COUNTER_car += 1
                        self.Cars[obj.id] = self.COUNTER_car
                        self.Cars.pop(obj.id)
                        print(
                            ">>" + str(obj.id) + " detected as a truck converted to car"
                        )

                    elif cont_m == "bus" and obj.counted_bus < 3:
                        obj.counted_bus += 1
                        print(
                            ">>"
                            + str(obj.id)
                            + " detected as a bus just set the counted_bus = "
                            + str(obj.counted_bus)
                        )

                if obj.id in self.Cars.keys():
                    if cont_m == "bus" and obj.counted_bus >= 3:
                        # this is probably a bus not a truck
                        self.COUNTER_car -= 1
                        self.COUNTER_bus += 1
                        self.Buses[obj.id] = self.COUNTER_bus
                        self.Cars.pop(obj.id)
                        print(
                            ">>" + str(obj.id) + " detected as a car converted to bus"
                        )

                    elif cont_m == "bus" and obj.counted_bus < 3:
                        obj.counted_bus += 1
                        print(
                            ">>"
                            + str(obj.id)
                            + " detected as a bus just set the counted_bus = "
                            + str(obj.counted_bus)
                        )

                if obj.id in self.Buses.keys():
                    if cont_m == "truck" and obj.counted_truck < 3:
                        obj.counted_truck += 1
                        print(
                            ">>"
                            + str(obj.id)
                            + " detected as a truck just set the counted_truck = "
                            + str(obj.counted_truck)
                        )

    def export_counter_initialization(self):

        header = ["Time"] + self.valid_selected_objects

        self.output_counter_file_name = self.output_counter_file_name + ".csv"

        if os.path.isfile(self.output_counter_file_name):
            os.remove(self.output_counter_file_name)

        with open(self.output_counter_file_name, "w", newline="") as csvfile:
            counters = csv.DictWriter(csvfile, fieldnames=header, lineterminator="\n")
            counters.writeheader()

    def export_counter_threading(self):
        self.counter_thread = threading.Thread(
            target=self.counter_export, args=(), daemon=True
        )
        self.counter_thread.start()

    def counter_export(self):

        header = ["Time"] + self.valid_selected_objects

        self.export_counter += 1

        ped_output_counter = 0
        cyclyst_output_counter = 0
        car_output_counter = 0
        truck_output_counter = 0
        bus_output_counter = 0
        cur_ped_counter = 0
        cur_cyclist_counter = 0
        cur_car_counter = 0
        cur_truck_counter = 0
        cur_bus_counter = 0

        for item in header:
            if item.lower() == "pedestrian":
                cur_ped_counter = self.COUNTER_p
                ped_output_counter = cur_ped_counter - self.last_exported_ped_counter
                if ped_output_counter < 0:
                    ped_output_counter = 0
            elif item.lower() == "cyclist":
                cur_cyclist_counter = self.COUNTER_c
                cyclyst_output_counter = (
                    cur_cyclist_counter - self.last_exported_cyclist_counter
                )
                if cyclyst_output_counter < 0:
                    cyclyst_output_counter = 0
            elif item.lower() == "car":
                cur_car_counter = self.COUNTER_car
                car_output_counter = cur_car_counter - self.last_exported_car_counter
                if car_output_counter < 0:
                    car_output_counter = 0
            elif item.lower() == "truck":
                cur_truck_counter = self.COUNTER_truck
                truck_output_counter = (
                    cur_truck_counter - self.last_exported_truck_counter
                )
                if truck_output_counter < 0:
                    truck_output_counter = 0
            elif item.lower() == "bus":
                cur_bus_counter = self.COUNTER_bus
                bus_output_counter = cur_bus_counter - self.last_exported_bus_counter
                if bus_output_counter < 0:
                    bus_output_counter = 0

        video_counted_minutes = config.periodic_counter_time * self.export_counter
        timedelta = datetime.timedelta(minutes=video_counted_minutes)

        with open(self.output_counter_file_name, "a+", newline="") as csvfile:
            counters = csv.DictWriter(csvfile, fieldnames=header, lineterminator="\n")
            data_object = {}

            for item in ["Time"] + self.valid_selected_objects:
                if item.lower() == "time":
                    data_object[item] = str(timedelta)
                elif item.lower() == "pedestrian":
                    data_object[item] = str(ped_output_counter)
                elif item.lower() == "cyclist":
                    data_object[item] = str(cyclyst_output_counter)
                elif item.lower() == "car":
                    data_object[item] = str(car_output_counter)
                elif item.lower() == "truck":
                    data_object[item] = str(truck_output_counter)
                elif item.lower() == "bus":
                    data_object[item] = str(bus_output_counter)

            data = [data_object]
            counters.writerows(data)

        self.last_exported_ped_counter = cur_ped_counter
        self.last_exported_cyclist_counter = cur_cyclist_counter
        self.last_exported_car_counter = cur_car_counter
        self.last_exported_truck_counter = cur_truck_counter
        self.last_exported_bus_counter = cur_bus_counter

        print("Counter exported to the csv file.........")
