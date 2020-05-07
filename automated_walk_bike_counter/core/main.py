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

from .tracking.object_tracker import ObjectTracker


# No motion detection
def count(self, bg):
    object_tracker = ObjectTracker()
    object_tracker.track_objects(self, self.FLAGS.demo, self.FLAGS.saveVideo)
