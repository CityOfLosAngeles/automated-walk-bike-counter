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

from .base import BaseModel


class MainModel(BaseModel):
    def __init__(self):
        super.__init__()

        self.allowed_objects = ["Person", "Cyclist", "Car", "Bus", "Truck"]
