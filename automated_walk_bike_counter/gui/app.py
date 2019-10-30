# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import tkinter as tk
from .controller.main_controller import MainController
from .view.main import MainView


class Application:
    def __init__(self, root=None, *args, **kwargs):
        if root is None:
            root = tk.Tk()
            root.title('CSULA Object Detection Project')
        self.root = root

        self.controller = MainController()
        self.view = MainView(root, self.controller)

    def run(self):
        self.root.geometry("1280x720+100+100")
        self.root.mainloop()

