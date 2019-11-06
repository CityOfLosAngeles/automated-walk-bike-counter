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

from tkinter import FALSE, Menu


class App_Menu(Menu):
    def __init__(self, parent, controller):

        parent.option_add("*tearOff", FALSE)
        menubar = Menu(parent)
        parent["menu"] = menubar

        menubar = Menu(parent)
        parent.config(menu=menubar)
        menu_file = Menu(menubar)
        menu_tools = Menu(menubar)
        menubar.add_cascade(menu=menu_file, label="File")
        menubar.add_cascade(menu=menu_tools, label="Tools")

        menu_file.add_command(label="Open...", command=controller.open_file)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=parent.quit)

        menu_tools.add_command(label="Add AOI", command=controller.add_new_aoi)
