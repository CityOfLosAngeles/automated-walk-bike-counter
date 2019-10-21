# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

import configparser
import utils.file_utils as fu
import os

class myConfiguration(object):

    def __init__(self, filename):
        self.config_options = {}
        self.parser = configparser.ConfigParser()
        self.parser.optionxform = str
        print("============> "+os.path.join(fu.get_project_root_dir(), filename))
        file = self.parser.read(os.path.join(fu.get_project_root_dir(), filename))

        if not file:
            raise ValueError('Config file not found in this address '+filename+'!')

        #print("Sections = "+str(self.parser.sections()))
        for name in self.parser.sections():
            for config_key,config_value in self.parser.items(name):
                print(config_key)
                print(config_value)
                self.config_options[config_key] = ( config_key , name )
            #self.__dict__.update(self.parser.items(name))
            
        self.init_config()

    def init_config(self):


        myConfiguration.PED_COST_THRESHOLD = self.parser.getint(self.config_options['PED_COST_THRESHOLD'][1] , self.config_options['PED_COST_THRESHOLD'][0])
        myConfiguration.BUS_COST_THRESHOLD = self.parser.getint(self.config_options['BUS_COST_THRESHOLD'][1] , self.config_options['BUS_COST_THRESHOLD'][0])
        myConfiguration.TRUCK_COST_THRESHOLD = self.parser.getint( self.config_options['TRUCK_COST_THRESHOLD'][1] , self.config_options['TRUCK_COST_THRESHOLD'][0] )

        myConfiguration.MISSING_THRESHOLD = self.parser.getint( self.config_options['MISSING_THRESHOLD'][1] , self.config_options['MISSING_THRESHOLD'][0] )
        myConfiguration.MISSING_THRESHOLD_MAX = self.parser.getint( self.config_options['MISSING_THRESHOLD_MAX'][1] , self.config_options['MISSING_THRESHOLD_MAX'][0] )

        myConfiguration.DUPLICATE_THRESHOLD = self.parser.getfloat( self.config_options['DUPLICATE_THRESHOLD'][1] , self.config_options['DUPLICATE_THRESHOLD'][0] )
        myConfiguration.BUS_TRUCK_DUPLICATE_THRESHOLD = self.parser.getfloat( self.config_options['BUS_TRUCK_DUPLICATE_THRESHOLD'][1] , self.config_options['BUS_TRUCK_DUPLICATE_THRESHOLD'][0] )
        myConfiguration.DUPLICATE_CAR_THRESHOLD = self.parser.getfloat( self.config_options['DUPLICATE_CAR_THRESHOLD'][1] , self.config_options['DUPLICATE_CAR_THRESHOLD'][0] )
        myConfiguration.DUPLICATE_TRUCK_THRESHOLD = self.parser.getfloat( self.config_options['DUPLICATE_TRUCK_THRESHOLD'][1] , self.config_options['DUPLICATE_TRUCK_THRESHOLD'][0] )
        myConfiguration.CAR_TRUCK_DUPLICATE_THRESHOLD = self.parser.getfloat( self.config_options['CAR_TRUCK_DUPLICATE_THRESHOLD'][1] , self.config_options['CAR_TRUCK_DUPLICATE_THRESHOLD'][0] )

        myConfiguration.COUNT_THRESHOLD = self.parser.getint(self.config_options['COUNT_THRESHOLD'][1] , self.config_options['COUNT_THRESHOLD'][0])
        myConfiguration.COUNT_THRESHOLD_BIKE = self.parser.getint(self.config_options['COUNT_THRESHOLD_BIKE'][1] , self.config_options['COUNT_THRESHOLD_BIKE'][0])
        myConfiguration.COUNT_THRESHOLD_MOTOR = self.parser.getint(self.config_options['COUNT_THRESHOLD_MOTOR'][1] , self.config_options['COUNT_THRESHOLD_MOTOR'][0])
        #myConfiguration.COUNT_THRESHOLD_CAR = self.parser.getint(self.config_options['COUNT_THRESHOLD_CAR'][1] , self.config_options['COUNT_THRESHOLD_CAR'][0])
        myConfiguration.COUNT_THRESHOLD_CAR = self.parser.getint(self.config_options['COUNT_THRESHOLD_CAR'][1] , self.config_options['COUNT_THRESHOLD_CAR'][0])
        myConfiguration.COUNT_THRESHOLD_BUS = self.parser.getint(self.config_options['COUNT_THRESHOLD_BUS'][1] , self.config_options['COUNT_THRESHOLD_BUS'][0])
        #myConfiguration.COUNT_THRESHOLD_TRUCK = self.parser.getint(self.config_options['COUNT_THRESHOLD_TRUCK'][1] , self.config_options['COUNT_THRESHOLD_TRUCK'][0])
        myConfiguration.COUNT_THRESHOLD_TRUCK = self.parser.getint(self.config_options['COUNT_THRESHOLD_TRUCK'][1] , self.config_options['COUNT_THRESHOLD_TRUCK'][0])

        myConfiguration.VALID_OBJECTS = self.parser.get(self.config_options['VALID_OBJECTS'][1] , self.config_options['VALID_OBJECTS'][0]).split(',')

config = myConfiguration(r'config.ini')
