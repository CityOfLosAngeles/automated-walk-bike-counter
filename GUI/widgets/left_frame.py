# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from tkinter import *
from tkinter.ttk import *
from tkinter.colorchooser import *
from tkinter import messagebox
from GUI.widgets.settings_pane import SettingsPane
from development.configuration import config


class Left_Frame(Frame):
    def __init__(self, parent, controller):
        super(Left_Frame, self).__init__(parent, height=600)
        self.style = Style().configure('TFrame.TFrame', background="yellow")
        self.controller = controller
        self.checkbox_variables = []
        self.color_objects = []
        self.allowed_objects = config.VALID_OBJECTS  # ['Person', 'Cyclist', 'Car', 'Truck', 'Bus']
        self.settings_pane = None
        self.create_objects_list_frame()

    def create_objects_list_frame(self):

        pane = LabelFrame(master=self,text="Valid Objects")
        pane.grid(row=0, column=0, padx=(5,5), pady=5, sticky=(W, E, S, N))

        self.grid_columnconfigure(0, weight=1)
        #parent.grid_rowconfigure(0, weight=1)

        for i in range(len(self.allowed_objects)):
            canvas = None
            self.checkbox_variables.append(IntVar())
            Checkbutton(pane, text = self.allowed_objects[i], variable= self.checkbox_variables[i],
                        command= lambda i=i : self.allowed_object_checkbox_changed(i) )\
                .grid(row=i, column=0, padx=(3, 3), pady=3, sticky=(W, E))

            self.color_objects.append(Canvas(master=pane, name=self.allowed_objects[i].lower(), width=20, height=20, bg='white'))
            self.color_objects[-1].grid(row=i, column=1, padx=(1,5), pady=3)
            self.color_objects[-1].bind("<Button-1>", lambda event, i=i: self.open_color_pallet_window(event,i))

            self.controller.valid_selected_objects.append(( self.allowed_objects[i], ((255, 255, 255), "#FFFFFF"), 0))

            pane.grid_rowconfigure(i, weight=1)

        pane.grid_columnconfigure(0, weight=1)
        pane.grid_columnconfigure(1, weight=0)

        self.settings_pane = SettingsPane(self,self.controller)
        self.settings_pane.grid(row=1, column=0, padx=(5, 5), pady=5, sticky=(W, E))

    def allowed_object_checkbox_changed(self,index):
        # messagebox.showinfo("title" , str(index)+" "+str(self.checkbox_variables[index].get())+" "+self.allowed_objects[index])
        tuple_data = self.controller.valid_selected_objects[index]
        list_data = list(tuple_data)
        list_data[-1] = self.checkbox_variables[index].get()
        self.controller.valid_selected_objects[index] = tuple(list_data)
        print(str(self.controller.valid_selected_objects))

    def open_color_pallet_window(self,event,index):
        color = askcolor()
        print(color)
        self.color_objects[index].configure(background=str(color[1]))
        #mb.showinfo("title"+str(color[1]))
        tuple_data = self.controller.valid_selected_objects[index]
        list_data = list(tuple_data)
        list_data[1] = color
        self.controller.valid_selected_objects[index] = tuple(list_data)
        print(str(self.controller.valid_selected_objects))


