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

import configargparse


def list_of_tuples(arg):
    return [tuple(map(int, x.split(","))) for x in arg.split(";")]


def list_of_items(arg):
    return [x for x in arg.split(",")]


parser = configargparse.get_argument_parser()

# Config file
parser.add_argument(
    "--config", is_config_file=True, help="Configuration file location", required=False,
)

# Cost thresholds
parser.add_argument(
    "--PED_COST_THRESHOLD",
    help="Pedestrian cost threshold",
    required=False,
    default=90,
    type=float,
    env_var="PED_COST_THRESHOLD",
)
parser.add_argument(
    "--BUS_COST_THRESHOLD",
    help="Bus cost threshold",
    default=110,
    required=False,
    type=float,
    env_var="BUS_COST_THRESHOLD",
)
parser.add_argument(
    "--TRUCK_COST_THRESHOLD",
    help="Truck cost threshold",
    default=110,
    required=False,
    type=float,
    env_var="TRUCK_COST_THRESHOLD",
)

# Missing thresholds
parser.add_argument(
    "--MISSING_THRESHOLD",
    help="Missing threshold",
    default=90,
    required=False,
    type=float,
    env_var="MISSING_THRESHOLD",
)
parser.add_argument(
    "--MISSING_THRESHOLD_MAX",
    help="Missing threshold maximum",
    default=300,
    required=False,
    type=float,
    env_var="MISSING_THRESHOLD_MAX",
)

# Duplicate thersholds
parser.add_argument(
    "--COUNT_THRESHOLD",
    help="Count threshold",
    default=7,
    required=False,
    type=int,
    env_var="COUNT_THRESHOLD",
)
parser.add_argument(
    "--COUNT_THRESHOLD_BIKE",
    help="Count threshold for bikes",
    default=3,
    required=False,
    type=int,
    env_var="COUNT_THRESHOLD_BIKE",
)
parser.add_argument(
    "--COUNT_THRESHOLD_MOTOR",
    help="Count threshold for...",
    default=3,
    required=False,
    type=int,
    env_var="COUNT_THRESHOLD_MOTOR",
)
parser.add_argument(
    "--COUNT_THRESHOLD_CAR",
    help="Count threshold for cars",
    default=3,
    required=False,
    type=int,
    env_var="COUNT_THRESHOLD_CAR",
)
parser.add_argument(
    "--COUNT_THRESHOLD_BUS",
    help="Count threshold for buses",
    default=3,
    required=False,
    type=int,
    env_var="COUNT_THRESHOLD_BUS",
)
parser.add_argument(
    "--COUNT_THRESHOLD_TRUCK",
    help="Count threshold for trucks",
    default=3,
    required=False,
    type=int,
    env_var="COUNT_THRESHOLD_TRUCK",
)

# Tracking settings
parser.add_argument(
    "--VALID_OBJECTS",
    nargs="+",
    help="List of valid objects",
    required=False,
    default=["Person", "Cyclist", "Car", "Truck", "Bus"],
    type=str,
    env_var="VALID_OBJECTS",
)

# Storage backend settings
parser.add_argument(
    "--AWS_REGION",
    help="AWS region for when the weights are stored in S3",
    required=False,
    type=str,
    default="us-west-1",
    env_var="AWS_REGION",
)

parser.add_argument(
    "--anchor_path",
    type=str,
    help="The path of the anchor txt file.",
    default=os.path.join(os.path.dirname(__file__), "data", "yolo_anchors.txt"),
)

parser.add_argument(
    "--new_size",
    nargs="*",
    type=int,
    default=[400, 400],
    help="Resize the input image with `new_size`, size format: [width, height]",
)

parser.add_argument(
    "--class_name_path",
    type=str,
    help="The path of the class names.",
    default=os.path.join(os.path.dirname(__file__), "data", "coco.names"),
)

parser.add_argument(
    "--restore_path",
    type=str,
    help="The path of the weights to restore.",
    default="s3://automated-walk-bike-counter/yolo/yolov3.ckpt",
    required=False,
)

parser.add_argument(
    "--save_video",
    type=lambda x: (str(x).lower() == "true"),
    default=True,
    help="Whether to save the video detection results.",
)

parser.add_argument(
    "--input_type",
    type=str,
    default="",
    choices=["file", "camera"],
    help="Whether the input is file or camera.",
)

parser.add_argument(
    "--file_name", type=str, help="The path of the input file.",
)

parser.add_argument(
    "--camera_id", type=int, default=0, help="The id of the input camera.",
)

parser.add_argument(
    "--search_objects",
    type=list_of_items,
    default=["Person", "Cyclist"],
    help="The list of valid objects comma separated like Person,Cyclist.",
)

parser.add_argument(
    "--objects_colors",
    type=list,
    default=["#ff0000", "#00ff00"],
    help="The list of objects' colors.",
)


parser.add_argument(
    "--aoi", type=list_of_tuples, help="The list of aoi coordinates.",
)

parser.add_argument(
    "--cli",
    type=lambda x: (str(x).lower() == "true"),
    help="Whether to run the application on cli.",
)

parser.add_argument(
    "--save_periodic_counter",
    type=lambda x: (str(x).lower() == "true"),
    help="Whether to save a periodic count",
)

parser.add_argument(
    "--periodic_counter_time",
    type=int,
    default=15,
    help="Interval (in minutes) to output the periodic counter",
)

config = parser.parse_known_args()[0]
