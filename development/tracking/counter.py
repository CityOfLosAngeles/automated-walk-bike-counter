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

    Pedestrians ={}
    Cyclists = {}
    Motorbikes = {}
    Duplicates = {}

    def __init__(self):    
        self.COUNTER_m = 0
        self.COUNTER_p = 0
        self.COUNTER_c = 0
        self.COUNTER_o = 0 
    

    def addNewMovingObjectForCounting(self,obj,position_new,postprocessed):
        cur_detected_object = obj.last_detected_object
        cont_m = cur_detected_object.mess
        ##for duplicated detection for bikers, when biker and motorbikers get detected as pedestrian first
        if (obj.id in self.Pedestrians.keys()):
            if(cont_m == 'bicycle' and obj.counted_biker >=5):
                ##this is probably a biker not a pedestrian
                self.COUNTER_p -= 1
                self.COUNTER_c += 1
                self.Cyclists[obj.id] = self.COUNTER_c
                self.Pedestrians.pop(obj.id)
    
            elif(cont_m == 'bicycle' and obj.counted_biker < 5):
                ##increase counter
                obj.counted_biker += 1
        
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
                cv2.putText(postprocessed,str(self.Pedestrians[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
            elif(cont_m == 'bicycle' and obj.counted >= self.COUNT_THRESHOLD_BIKE):
                ##ever detected as pedestrian, added 4/18 for prevent detecting bicycle without rider
                if(obj.pedestrian_id ==1):
                    print("counted bicycle " + str(obj.id) + " " + str(obj.counted))

                    self.COUNTER_c += 1
                    self.Cyclists[obj.id] = self.COUNTER_c
                    ## mark the moving object with the id
                    cv2.putText(postprocessed,str(self.Cyclists[obj.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
            ##added on 7/23
            elif(cont_m == 'motorbike' and obj.counted >= self.COUNT_THRESHOLD_MOTOR):
                print("counted motorbike " + str(obj.id) + " " + str(obj.counted))
                self.COUNTER_o += 1
                self.Motorbikes[obj.id] = self.COUNTER_o
