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

"""
Automated Walk Bike Counter

A computer vision application for automating the detection of pedestrians,
cyclists, and cars from traffic cameras.
"""
import logging
import warnings

from ._version import __version__  # noqa: F401

# Suppress some annoying warnings from Tensorflow
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
del logging, warnings
