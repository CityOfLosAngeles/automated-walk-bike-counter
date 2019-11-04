# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
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
import configparser

from ..utils import file_utils as fu

parser = configargparse.get_argument_parser()

# Config file
default_filename = os.path.join(os.path.dirname(__file__), "config.ini")
parser.add_argument(
    "--config",
    is_config_file=True,
    help="Configuration file location",
    default=default_filename,
    required=False,
)

# Cost thresholds
parser.add_argument(
    "--PED_COST_THRESHOLD",
    help="Pedestrian cost threshold",
    required=True,
    type=float,
    env_var="PED_COST_THRESHOLD",
)
parser.add_argument(
    "--BUS_COST_THRESHOLD",
    help="Bus cost threshold",
    required=True,
    type=float,
    env_var="BUS_COST_THRESHOLD",
)
parser.add_argument(
    "--TRUCK_COST_THRESHOLD",
    help="Truck cost threshold",
    required=True,
    type=float,
    env_var="TRUCK_COST_THRESHOLD",
)

# Missing thresholds
parser.add_argument(
    "--MISSING_THRESHOLD",
    help="Missing threshold",
    required=True,
    type=float,
    env_var="MISSING_THRESHOLD",
)
parser.add_argument(
    "--MISSING_THRESHOLD_MAX",
    help="Missing threshold maximum",
    required=True,
    type=float,
    env_var="MISSING_THRESHOLD_MAX",
)

# Duplicate thersholds
parser.add_argument(
    "--COUNT_THRESHOLD",
    help="Count threshold",
    required=True,
    type=int,
    env_var="COUNT_THRESHOLD",
)
parser.add_argument(
    "--COUNT_THRESHOLD_BIKE",
    help="Count threshold for bikes",
    required=True,
    type=int,
    env_var="COUNT_THRESHOLD_BIKE",
)
parser.add_argument(
    "--COUNT_THRESHOLD_MOTOR",
    help="Count threshold for...",
    required=True,
    type=int,
    env_var="COUNT_THRESHOLD_MOTOR",
)
parser.add_argument(
    "--COUNT_THRESHOLD_CAR",
    help="Count threshold for cars",
    required=True,
    type=int,
    env_var="COUNT_THRESHOLD_CAR",
)
parser.add_argument(
    "--COUNT_THRESHOLD_BUS",
    help="Count threshold for buses",
    required=True,
    type=int,
    env_var="COUNT_THRESHOLD_BUS",
)
parser.add_argument(
    "--COUNT_THRESHOLD_TRUCK",
    help="Count threshold for trucks",
    required=True,
    type=int,
    env_var="COUNT_THRESHOLD_TRUCK",
)

# Tracking settings
parser.add_argument(
    "--VALID_OBJECTS",
    help="Count threshold for...",
    action="append",
    required=True,
    type=str,
    env_var="VALID_OBJECTS",
)

config = parser.parse_known_args()[0]
