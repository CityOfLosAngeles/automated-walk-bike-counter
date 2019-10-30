# Copyright (c) Data Science Research Lab at California State University Los Angeles (CSULA), and City of Los Angeles ITA
# Distributed under the terms of the Apache 2.0 License
# www.calstatela.edu/research/data-science
# Designed and developed by:
# Data Science Research Lab
# California State University Los Angeles
# Dr. Mohammad Pourhomayoun
# Mohammad Vahedi
# Haiyan Wang

from ..movingobject import MovingObject
import cv2

class Object_Counter:

    #change to 10 from 20 on 2/26 because biker doesn't get counted correctly
    COUNT_THRESHOLD = 10

    COUNT_THRESHOLD_BIKE=5
    COUNT_THRESHOLD_MOTOR = 3
    #COUNT_THRESHOLD_CAR = 6
    COUNT_THRESHOLD_CAR = 5
    COUNT_THRESHOLD_BUS = 5
    #COUNT_THRESHOLD_TRUCK = 6
    COUNT_THRESHOLD_TRUCK = 5

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
        self.Pedestrians ={}
        self.Cyclists = {}
        self.Trucks = {}
    

    def addNewMovingObjectForCounting(self,obj,position_new,postprocessed):
        cur_detected_object = obj.last_detected_object
        cont_m = cur_detected_object.mess

        print("Counting for object id:"+str(obj.id)+" as "+cont_m)
    
        ##for duplicated detection for bikers, when biker and motorbikers get detected as pedestrian first
        if (cont_m=='person' or cont_m == 'bicycle' or cont_m == 'motorbike'):

            if (obj.id in self.Pedestrians.keys()):

                if(cont_m == 'bicycle' and obj.counted_biker >=3):
                    ##this is probably a biker not a pedestrian
                    self.COUNTER_p -= 1
                    self.COUNTER_c += 1
                    self.Cyclists[obj.id] = self.COUNTER_c
                    self.Pedestrians.pop(obj.id)
                    print("Person "+str(obj.id)+" counted for equal or more than 3 times for bicycle and detected as cyclist")
        
                elif(cont_m == 'bicycle' and obj.counted_biker < 3):
                    ##increase counter
                    obj.counted_biker += 1
                    print("Person "+str(obj.id)+" counted for bicycle for "+str( obj.counted_biker)+" times.")
            
                if(cont_m == 'motorbike' and obj.counted_moter >=3):
                    ##this is probably a moterbiker
                    self.COUNTER_p -= 1
                    self.COUNTER_o += 1
                    self.Motorbikes[obj.id] = self.COUNTER_o
                    self.Pedestrians.pop(obj.id)

                elif(cont_m == 'motorbike' and obj.counted_moter <3):
                    obj.counted_moter += 1


            if (not obj.id in self.Pedestrians.keys() and not obj.id in self.Cyclists.keys() and not obj.id in self.Motorbikes.keys()):

                if(cont_m == 'person' and obj.counted >= self.COUNT_THRESHOLD):
                    print("counted person " + str(obj.id) + " " + str(obj.counted))
                    (position_x,position_y) = obj.position[-1] 
                    self.COUNTER_p += 1
                    self.Pedestrians[obj.id] = self.COUNTER_p
                    obj.pedestrian_id = 1
                    ## mark the moving object with the id
                    #cv2.putText(postprocessed,str(self.Pedestrians[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
                elif(cont_m == 'bicycle' and obj.counted >= self.COUNT_THRESHOLD_BIKE):
                    ##ever detected as pedestrian, added 4/18 for prevent detecting bicycle without rider
                    if(obj.pedestrian_id ==1):
                        print("counted bicycle " + str(obj.id) + " " + str(obj.counted))

                        self.COUNTER_c += 1
                        self.Cyclists[obj.id] = self.COUNTER_c
                        ## mark the moving object with the id
                        #cv2.putText(postprocessed,str(self.Cyclists[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
                ##added on 7/23
                elif(cont_m == 'motorbike' and obj.counted >= self.COUNT_THRESHOLD_MOTOR):
                    print("counted motorbike " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_o += 1
                    self.Motorbikes[obj.id] = self.COUNTER_o
        else:

            if ((not obj.id in self.Cars.keys()) and (not obj.id in self.Buses.keys()) and (not obj.id in self.Trucks.keys())):
                if (cont_m == 'car' and obj.counted >= self.COUNT_THRESHOLD_CAR):
                    print("counted car " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_car += 1
                    self.Cars[obj.id] = self.COUNTER_car
                    print(">>"+str(obj.id)+" detected as new car")
                    
                    #cv2.putText(postprocessed,str(self.Cars[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)

                elif (cont_m == 'bus' and obj.counted >= self.COUNT_THRESHOLD_BUS):
                    print("counted bus " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_bus += 1
                    self.Buses[obj.id] = self.COUNTER_bus
                    print(">>"+str(obj.id)+" detected as new bus")
                    #cv2.putText(postprocessed,str(self.Buses[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)

                elif (cont_m == 'truck' and obj.counted >= self.COUNT_THRESHOLD_TRUCK):
                    print("counted truck " + str(obj.id) + " " + str(obj.counted))
                    self.COUNTER_truck += 1
                    self.Trucks[obj.id] = self.COUNTER_truck
                    print(">>"+str(obj.id)+" detected as new truck")
                    #cv2.putText(postprocessed,str(self.Trucks[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
            else:
                
                if (obj.id in self.Trucks.keys()):
                    if(cont_m == 'bus' and obj.counted_bus >=3):
                        ##this is probably a bus not a truck
                        self.COUNTER_truck -= 1
                        self.COUNTER_bus += 1
                        self.Buses[obj.id] = self.COUNTER_bus
                        self.Trucks.pop(obj.id)
                        print(">>"+str(obj.id)+" detected as a truck converted to bus")
                    
                    elif(cont_m == 'car' and obj.counted_car >=3):
                        ##this is probably a bus not a truck
                        self.COUNTER_truck -= 1
                        self.COUNTER_car += 1
                        self.Cars[obj.id] = self.COUNTER_car
                        self.Cars.pop(obj.id)
                        print(">>"+str(obj.id)+" detected as a truck converted to car")

                    elif(cont_m == 'bus' and obj.counted_bus < 3):
                        obj.counted_bus += 1
                        print(">>"+str(obj.id)+" detected as a bus just set the counted_bus = "+str(obj.counted_bus))

                if (obj.id in self.Cars.keys()):
                    if(cont_m == 'bus' and obj.counted_bus >=3):
                        ##this is probably a bus not a truck
                        self.COUNTER_car -= 1
                        self.COUNTER_bus += 1
                        self.Buses[obj.id] = self.COUNTER_bus
                        self.Cars.pop(obj.id)
                        print(">>"+str(obj.id)+" detected as a car converted to bus")

                    elif(cont_m == 'bus' and obj.counted_bus < 3):
                        obj.counted_bus += 1
                        print(">>"+str(obj.id)+" detected as a bus just set the counted_bus = "+str(obj.counted_bus))

                if (obj.id in self.Buses.keys()):
                    if(cont_m == 'truck' and obj.counted_truck < 3):
                        obj.counted_truck += 1
                        print(">>"+str(obj.id)+" detected as a truck just set the counted_truck = "+str(obj.counted_truck))

            
