"""
tfnet secondary (helper) methods
"""
from ..utils.loader import create_loader
from time import time as timer
import tensorflow as tf
import numpy as np
import sys
import cv2
import os
import re
import math
from munkres import Munkres,print_matrix
from pykalman import KalmanFilter
import sys
sys.path.append('C:/Users/Data Science/Documents/darkflow/track')
from movingobject import MovingObject


old_graph_msg = 'Resolving old graph def {} (no guarantee)'

def build_train_op(self):
    self.framework.loss(self.out)
    self.say('Building {} train op'.format(self.meta['model']))
    optimizer = self._TRAINER[self.FLAGS.trainer](self.FLAGS.lr)
    gradients = optimizer.compute_gradients(self.framework.loss)
    self.train_op = optimizer.apply_gradients(gradients)

def load_from_ckpt(self):
    if self.FLAGS.load < 0: # load lastest ckpt
        with open(self.FLAGS.backup + 'checkpoint', 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.FLAGS.load = int(load_point)
    
    load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
    load_point = '{}-{}'.format(load_point, self.FLAGS.load)
    self.say('Loading from {}'.format(load_point))
    try: self.saver.restore(self.sess, load_point)
    except: load_old_graph(self, load_point)

def say(self, *msgs):
    if not self.FLAGS.verbalise:
        return
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        print(msg)

def load_old_graph(self, ckpt): 
    ckpt_loader = create_loader(ckpt)
    self.say(old_graph_msg.format(ckpt))
    
    for var in tf.global_variables():
        name = var.name.split(':')[0]
        args = [name, var.get_shape()]
        val = ckpt_loader(args)
        assert val is not None, \
        'Cannot find and load {}'.format(var.name)
        shp = val.shape
        plh = tf.placeholder(tf.float32, shp)
        op = tf.assign(var, plh)
        self.sess.run(op, {plh: val})

def _get_fps(self, frame):
    elapsed = int()
    start = timer()
    preprocessed = self.framework.preprocess(frame)
    feed_dict = {self.inp: [preprocessed]}
    net_out = self.sess.run(self.out, feed_dict)[0]
    processed = self.framework.postprocess(net_out, frame, False)
    return timer() - start

def camera(self):
    file = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo
    
    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), \
        'file {} does not exist'.format(file)
        
    camera = cv2.VideoCapture(file)
    
    if file == 0:
        self.say('Press [ESC] to quit demo')
        
    assert camera.isOpened(), \
    'Cannot capture source'
    
    if file == 0:#camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('', width, height)
    else:
        _, frame = camera.read()
        height, width, _ = frame.shape

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print("video : ", file)
        m = re.match(r"([^\.]*)(\..*)",file)
        vfname = m.group(1)
        outfile = vfname + "_result.avi"
        print(vfname)
        if file == 0:#camera window
          fps = 1 / self._get_fps(frame)
          if fps < 1:
            fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        #videoWriter = cv2.VideoWriter(
            #'video.avi', fourcc, fps, (width, height))
        videoWriter = cv2.VideoWriter(
            outfile, fourcc, fps, (width, height))
    
    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()
    
    elapsed = int()
    start = timer()
    self.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break
        preprocessed = self.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            print("elapsed",elapsed)
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                (postprocessed, boxes) = self.framework.postprocess(
                    single_out, img, False)
                print("boxes", len(boxes))
                for box in boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = box
                    print("box",left, right, top, bot, mess, max_indx, confidence)
                if SaveVideo:
                    videoWriter.write(postprocessed)
                if file == 0: #camera window
                    cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if file == 0: #camera window
            choice = cv2.waitKey(1)
            if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    if file == 0: #camera window
        cv2.destroyAllWindows()
        

##No motion detection
def count(self, bg):
    file = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo
    savebg = bg
    print("video : ", file)
    m = re.match(r"([^\.]*)(\..*)",file)
    vfname = m.group(1)
    
    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), \
        'file {} does not exist'.format(file)
        
    camera = cv2.VideoCapture(file)
    
    if file == 0:
        self.say('Press [ESC] to quit demo')
        
    assert camera.isOpened(), \
    'Cannot capture source'
    
    if file == 0:#camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('', width, height)
    else:
        _, frame = camera.read()
        ##if resize:
        ##    resize(frame, w=width_new)
        height, width, _ = frame.shape

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outfile = vfname + "_result.mp4"
        print(outfile)
        if file == 0:#camera window
          fps = 1 / self._get_fps(frame)
          if fps < 1:
            fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        #videoWriter = cv2.VideoWriter(
            #'video.avi', fourcc, fps, (width, height))
        videoWriter = cv2.VideoWriter(
            outfile, fourcc, fps, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()
    
    elapsed = int()
    start = timer()
    self.say('Press [ESC] to quit demo')
    
    ########################################################
    ##inner functions
    def overlap_area(boxes):
        if(len(boxes) == 0):
            return 0

        xx1 = max(boxes[0,0], boxes[1,0])
        yy1 = max(boxes[0,1], boxes[1,1])
        xx2 = min(boxes[0,2], boxes[1,2]) 
        yy2 = min(boxes[0,3], boxes[1,3])

        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)

        ##box1 area
        area1 = (boxes[0,2]-boxes[0,0]) * (boxes[0,3]-boxes[0,1])
        ##box2 area
        area2 = (boxes[1,2]-boxes[1,0]) * (boxes[1,3]-boxes[1,1])

        area = min(area1,area2)    
        overlap = (w * h) / area

        return overlap

    ##calculate distance between 2 points, pos: [0,0], points: [[1,1],[2,2],[3,3]]
    def get_costs(pos,points):
        distances = [math.floor(math.sqrt((x2-pos[0])**2+(y2-pos[1])**2)) for (x2,y2) in points]
        return distances

        ###for logging
    def mylogger(name, logfile):
        logger = logging.getLogger(name)
        if not len(logger.handlers):
            logger = logging.getLogger(name)
            hdlr = logging.FileHandler(logfile)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
            logger.setLevel(logging.INFO)

        return logger

    def update_skipped_frame(frame,fname,tracks,thresh):

        h,w = frame.shape[:2]
        counter = "Pedestrians: " +  str(COUNTER_p) + " Cyclists: "+ str(COUNTER_c) + " frame: " + str(elapsed)
        cv2.putText(frame,counter,(20,int(h-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
        ##update tracking objects
        for obj in tracks:
            obj.frames_since_seen += 1
        new_tracks = removeTrackedObjects(tracks,frame,thresh)
        print("update skipped frame")

        return new_tracks

    def removeTrackedObjects(tracking_arr,frame,thresh):
        #MISSING_THREASHOLD = 90
        for index, obj in enumerate(tracking_arr):
        ##if a moving object hasn't been updated for 10 frames then remove it
            if obj.frames_since_seen > thresh:
                del tracking_arr[index]
            ## if the object is out of the scene then remove from current tracking right away
            h,w = frame.shape[:2]
            if (obj.position[-1][0] < 0 or obj.position[-1][0] > w):
                del tracking_arr[index]
            elif (obj.position[-1][1] < 0 or obj.position[-1][1] > h):
                del tracking_arr[index]

        return tracking_arr
    
    def track_new_object(position, tracks, counter):
        new_mObject = MovingObject(counter,position)
        new_mObject.add_position([position])
        new_mObject.init_kalman_filter()
        filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
        new_mObject.set_next_mean(filtered_state_means[-1])
        new_mObject.set_next_covariance(filtered_state_covariances[-1])
        ##add to current_tracks
        tracks.append(new_mObject)
    ######################################################################
    
    ####################################
    ##motion detect and count
    POSITION_THREASHOLD= 30
    HIGHT_THRESHOLD = 50
    ##missing frame numbers to dismiss the object from the current_tracks
    MISSING_THREASHOLD = 90
    MISSING_THREASHOLD_MAX = 300
    ##minimum contour area to consider
    #CONTOUR_THREASHOLD_MIN = 300
    CONTOUR_THREASHOLD_MIN = 400
    ##maximum contour area to consider
    CONTOUR_THREASHOLD_MAX = 8000
    #CONTOUR_THREASHOLD_MAX = 5000
    PED_AREA_THRESHOLD = 600
    BIK_AREA_THRESHOLD = 700

    #change on 3/29 for testing purpose
    COST_THREASHOLD = 80
    #COST_THREASHOLD = 140
    
    BOUNDRY = 30

    ##pedestrian counter  ##added COUNTER_o for counting motorbikes 7/27
    COUNTER_m = 0
    COUNTER_p = 0
    COUNTER_c = 0
    COUNTER_o = 0
    ##adjustment for pedestrian and cyclist counting duplication
    COUNTER_p_adj = 0
    COUNTER_c_adj = 0
    OVERLAP_THREASHOLD = 0.50
    OVERLAP_TIMES_THRESHOLD = 10
    DISTANCE_THREASHOLD = 20
    #change to 10 from 20 on 2/26 because biker doesn't get counted correctly
    COUNT_THRESHOLD = 10
    COUNT_THRESHOLD_BIKE=5
    COUNT_THRESHOLD_MOTOR = 3
    
    ##duplicated classification threshold > 0.30 then means 2 classification for same object
    DUPLICATE_THREASHOLD = 0.30

    ##init array to hold current objects
    current_tracks = []

    ##debug
    debug = True 
    PEDESTRIAN_DEBUG = True
    SAVE_FRAME = False

    Pedestrians ={}
    Cyclists = {}
    Motorbikes = {}
    Duplicates = {}
    n = 0
    avg = None

    ##get the input video file name without extension
    ## ex. Camera65_13_00_09272017_0
    print("video : ", file)
    m = re.match(r"([^\.]*)(\..*)",file)
    vfname = m.group(1)
    print("vfname : "+vfname)

    ##background
    QUICK_BACKGROUND_SAVE = False 
    if savebg == 1:
        QUICK_BACKGROUND_SAVE = True
    background = 'background/' + vfname + '_bg.npy'
    if os.path.exists(background):
        avg = np.load(background)

    ####################################
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break

		
        preprocessed = self.framework.preprocess(frame)
        #print("preprocessed dimension >>>>>>>>> : "+str(preprocessed.shape))         # ---- Vahedi
        #print("preprocessed dimension -------------- : "+str(frame.shape))           # ---- Vahedi
        #if (elapsed==100):                                                           # ---- Vahedi
        #    cv2.imwrite("C:\\Users\\Data Science\\Documents\\temp\\1.png", frame           # ---- Vahedi
        #    cv2.imwrite("C:\\Users\\Data Science\\Documents\\temp\\2.png", preprocessed)   # ---- Vahedi
			
        #print("preprocessed frame no."+str(elapsed)+" : "+str(preprocessed))         # ---- Vahedi
        #print("frame no. : "+str(elapsed)+" : "+str(frame))                          # ---- Vahedi
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            #print("self.inp : "+ str(self.inp))                  # ---- Vahedi
            #print("feed_dict : "+ str(feed_dict))                  # ---- Vahedi
            net_out = self.sess.run(self.out, feed_dict)
            #print("net_out : "+ str(net_out))                  # ---- Vahedi
            print("Frame: ", elapsed)
            for img, single_out in zip(buffer_inp, net_out):
                #if (elapsed==337):                                                          # ---- Vahedi
                #    cv2.imwrite("C:\\Users\\Data Science\\Documents\\temp\\10.png", img)    # ---- Vahedi
                #    print("net_out : ====== ", single_out)                                  # ---- Vahedi

                (postprocessed, boxes) = self.framework.postprocess(single_out, img, False)
                if (elapsed==1):                                                          # ---- Vahedi
                     cv2.imwrite("C:\\Users\\Data Science\\Documents\\temp\\20.png", postprocessed)  # ---- Vahedi

                for box in boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = box
                    print("box",left, right, top, bot, mess, max_indx, confidence)
                #########################################
                
                #if(len(current_tracks) > 0):
                #    for index, obj in enumerate(current_tracks):
                #        obj.counted = 0
                
                ##seperate pedestrian and cyclists (because both have person)
                ped_boxes = []
                bik_boxes = []
                mot_boxes = []
                nodup_boxes = []
                ##duplicated counting for cylist
                ped_boxes_dup_dict = {}
                for box in boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = box
                    if(mess == 'person'):
                        ped_boxes.append(box)
                    elif(mess == 'bicycle'):
                        bik_boxes.append(box)
                    elif(mess == 'motorbike'):
                        mot_boxes.append(box)
                ##compare bike/motorbike and pedestrian, find duplicates
                for bik in bik_boxes:
                    (bleft, bright, btop, bbot, bmess, bmax_indx, bconfidence) = bik
                    for ped in ped_boxes:
                        (left, right, top, bot, mess, max_indx, confidence) = ped
                        boxes_2compare = np.array([[left,top,right,bot],[bleft,btop,bright,bbot]])
                        o_rate = overlap_area(boxes_2compare)
                        print("overlap: ", o_rate)
                        if(o_rate > DUPLICATE_THREASHOLD):
                            ped_boxes_dup_dict[ped] =1
                            print("exclude for duplicates:",left, right, top, bot )
                for mot in mot_boxes:
                    (bleft, bright, btop, bbot, bmess, bmax_indx, bconfidence) = mot
                    for ped in ped_boxes:
                        (left, right, top, bot, mess, max_indx, confidence) = ped
                        boxes_2compare = np.array([[left,top,right,bot],[bleft,btop,bright,bbot]])
                        o_rate = overlap_area(boxes_2compare)
                        print("overlap: ", o_rate)
                        if(o_rate > DUPLICATE_THREASHOLD):
                            ped_boxes_dup_dict[ped] =1
                            print("exclude for duplicates:",left, right, top, bot )
                ##add pedestrian only into nodup_boxes
                for pbox in ped_boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = pbox                    
                    if pbox in ped_boxes_dup_dict:
                        continue
                    else:
                        nodup_boxes.append(pbox)
                
                ##add bike into nodup_boxes
                for bbox in bik_boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = bbox
                    ##add filter to exclude bicycle bbox too small, could be a false detection of a handbag
                    if((right-left)*(bot-top) < BIK_AREA_THRESHOLD and (bot-top) < (right-left)*1.5 ):
                        continue
                    else:
                        nodup_boxes.append(bbox)
                
                ##add motorbikers 7/27, since we need to do better job for excluding motorbikers
                for mbox in mot_boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = mbox
                    nodup_boxes.append(mbox)
                
                insider_dict = {}
                noins_boxes = []
                margin = 10
                ##remove any box that inside of other boxes
                for nbox in nodup_boxes:
                    for mbox in nodup_boxes:
                        if(nbox == mbox):
                            continue
                        (nleft, nright, ntop, nbot, nmess, nmax_indx, nconfidence) = nbox
                        (mleft, mright, mtop, mbot, mmess, mmax_indx, mconfidence) = mbox
                        if(mleft+margin >= nleft and mright-margin <= nright and mtop+margin>=ntop and mbot-margin<=nbot):
                            insider_dict[mbox] = 1
                            print(mleft, mright, mtop, mbot," inside of ", nleft, nright, ntop, nbot)
                            
                for fbox in nodup_boxes:
                    if fbox in insider_dict:
                        continue
                    else:
                        noins_boxes.append(fbox)
                
                contours = np.zeros((0,2))
                contours_orig = np.zeros((0,5))
                for box in noins_boxes:
                    (left, right, top, bot, mess, max_indx, confidence) = box
                    ##04/03/2018 exclude pedestrian inside of car by bbox size and shape
                    if(mess=='person' and (((right-left)*(bot-top) < PED_AREA_THRESHOLD) or (bot-top) < (right-left) * 1.5)):
                        continue
                    
                    cx = left + int((right-left)/2)
                    cy = top + int((bot-top)/2)
                    tdata = [cx,cy]
                    contours_orig = np.append(contours_orig,[[left,top,right,bot,mess]],axis=0)
                    contours = np.append(contours, [tdata],axis=0)
                
                ##first moving object
                if(len(current_tracks) == 0):
                    #for cont in contours:
                    for box in noins_boxes:
                        (left, right, top, bot, mess, max_indx, confidence) = box
                        print("Box : ===>>> "+str(box))
                        #print("Contours : "+str(cont))
                        cx = left + int((right-left)/2)
                        cy = top + int((bot-top)/2)
                        tdata = [cx,cy]
                        COUNTER_m += 1
                        ##create new movingObject
                        new_mObject = MovingObject(COUNTER_m,tdata)
                        new_mObject.add_position([tdata])
                        new_mObject.init_kalman_filter()
                        filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
                        print("filtered_state_means : "+str(filtered_state_means))                       #  -------   Vahedi
                        print("filtered_state_covariances : "+str(filtered_state_covariances))           #  -------   Vahedi
                        new_mObject.set_next_mean(filtered_state_means[-1])
                        new_mObject.set_next_covariance(filtered_state_covariances[-1])
                        new_mObject.counted += 1

                        ##add to current_tracks
                        current_tracks.append(new_mObject)
                        print("current_tracks objects : ")  
						#  -------   Vahedi
                    print('\n'.join(str(ct) for ct in current_tracks))                  #  -------   Vahedi
                        
                ##from the 2nd frame, calculate cost using predicted position and new contour positions
                else:
                    ##save all the positions to a matrix and calculate the cost
                    ##initiate a matrix for calculating assianment by Hungarian algorithm
                    ##when no contour found in this frame then update kalman filter and skip
                    if(len(contours) == 0):
                        n = n + 1
                        current_tracks = update_skipped_frame(postprocessed,elapsed,current_tracks,MISSING_THREASHOLD)
                        #counter = "Pedestrians: " +  str(COUNTER_p) + " Cyclists: "+ str(COUNTER_c) + " frame: " + str(elapsed)
                        #cv2.putText(postprocessed,counter,(20,int(h-30)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
                        if SaveVideo:
                            videoWriter.write(postprocessed)
                        #if(SAVE_FRAME):
                            #savepath = "./images/"
                            #cv2.imwrite(savepath, postprocessed)
                        continue
                    matrix_h =[]
                    remove_obj = []    
                    
                    available_tracks = []
                    for index, obj in enumerate(current_tracks):
                        ##calculate costs for each tracked movingObjects using their predicted position
                        costs = get_costs(obj.predicted_position[-1], contours)

                        ## if tracking object to all contours distances are too large, then not to consider it at all
                        if all(c > COST_THREASHOLD for c in costs):
                            ##update it with KF predicted position
                            obj.kalman_update_missing(obj.predicted_position[-1])
                            ##skip this tracking object
                            continue

                        matrix_h.append(costs)
                        ##only valid tracking objects are added to available_tracks
                        available_tracks.append(obj)
                    
                    munkres = Munkres()
                    ## when matrix is empty, skip this frame
                    if(len(matrix_h) < 1):
                        n = n + 1
                        current_tracks = update_skipped_frame(postprocessed,elapsed,current_tracks,MISSING_THREASHOLD)
                        #counter = "Pedestrians: " +  str(COUNTER_p) + " Cyclists: "+ str(COUNTER_c) + " frame: " + str(elapsed)
                        #cv2.putText(postprocessed,counter,(20,int(h-30)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
                        if SaveVideo:
                            videoWriter.write(postprocessed)
                        #if(SAVE_FRAME):
                            #savepath = "./images/"
                            #cv2.imwrite(savepath, postprocessed)
                        continue
                    indexes = munkres.compute(matrix_h)

                    total = 0
                    for row, column in indexes:
                        value = matrix_h[row][column]
                        total += value
                        
                    ## loop through the contours and update Kalman filter for each contour
                    indexes_np = np.array(indexes)
                    for index_c,cont in enumerate(contours):
                        ## found contour index, then update this contour position with the tracked object
                        if index_c in indexes_np[:,1]:
                            contour_index_list = indexes_np[:,1].tolist()
                            index_m = contour_index_list.index(index_c)
                            ##find index in current_tracks
                            index_track = indexes_np[index_m,0]
                            
                            ##check if cost is bigger than threashold then track it as a new one
                            if matrix_h[index_track][index_c] > COST_THREASHOLD:
                                
                                ##create new MovingObject of this contour
                                COUNTER_m += 1
                                track_new_object(cont, current_tracks, COUNTER_m)
                                continue

                            ##get the object from the index and update it's Kalman filter
                            obj_m = available_tracks[index_track]
                            ##get corresponding contour position, update kalman filter
                            position_new = contours[index_c]
                            obj_m.kalman_update(position_new)
                            obj_m.counted += 1
                            print("counted " + str(obj_m.id) + " " + str(obj_m.counted))
                            ##get the original contour x,y,w,h, not the center corrdination
                            cont_x,cont_y,cont_w,cont_h,cont_m = contours_orig[index_c]
                            
                            ##for duplicated detection for bikers, when biker and motorbikers get detected as pedestrian first
                            if (obj_m.id in Pedestrians.keys()):
                                if(cont_m == 'bicycle' and obj_m.counted_biker >=5):
                                    ##this is probably a biker not a pedestrian
                                    COUNTER_p -= 1
                                    COUNTER_c += 1
                                    Cyclists[obj_m.id] = COUNTER_c
                                    Pedestrians.pop(obj_m.id)
                                elif(cont_m == 'bicycle' and obj_m.counted_biker < 5):
                                    ##increase counter
                                    obj_m.counted_biker += 1
                                    
                                if(cont_m == 'motorbike' and obj_m.counted_moter >=3):
                                    ##this is probably a moterbiker
                                    COUNTER_p -= 1
                                    COUNTER_o += 1
                                    Motorbikes[obj_m.id] = COUNTER_o
                                    Pedestrians.pop(obj_m.id)
                                elif(cont_m == 'motorbike' and obj_m.counted_moter <3):
                                    obj_m.counted_moter += 1
                            
                            if (not obj_m.id in Pedestrians.keys() and not obj_m.id in Cyclists.keys() and not obj_m.id in Motorbikes.keys()):
                                if(cont_m == 'person' and obj_m.counted >= COUNT_THRESHOLD):
                                    print("counted person " + str(obj_m.id) + " " + str(obj_m.counted))
                                    (position_x,position_y) = obj_m.position[-1] 
                                    COUNTER_p += 1
                                    Pedestrians[obj_m.id] = COUNTER_p
                                    obj_m.pedestrian_id = 1
                                    ## mark the moving object with the id
                                    cv2.putText(postprocessed,str(Pedestrians[obj_m.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
                                elif(cont_m == 'bicycle' and obj_m.counted >= COUNT_THRESHOLD_BIKE):
                                    ##ever detected as pedestrian, added 4/18 for prevent detecting bicycle without rider
                                    if(obj_m.pedestrian_id ==1):
                                        print("counted bicycle " + str(obj_m.id) + " " + str(obj_m.counted))
                                        COUNTER_c += 1
                                        Cyclists[obj_m.id] = COUNTER_c
                                        ## mark the moving object with the id
                                        cv2.putText(postprocessed,str(Cyclists[obj_m.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)
                                ##added on 7/23
                                elif(cont_m == 'motorbike' and obj_m.counted >= COUNT_THRESHOLD_MOTOR):
                                    print("counted motorbike " + str(obj_m.id) + " " + str(obj_m.counted))
                                    COUNTER_o += 1
                                    Motorbikes[obj_m.id] = COUNTER_o
                                            
                        else:
                            position_new = contours[index_c]
                            COUNTER_m += 1
                            new_mObject = MovingObject(COUNTER_m,position_new)
                            new_mObject.add_position([position_new])
                            new_mObject.init_kalman_filter()
                            
                            filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
                            new_mObject.set_next_mean(filtered_state_means[-1])
                            new_mObject.set_next_covariance(filtered_state_covariances[-1])

                            ##add to current_tracks
                            current_tracks.append(new_mObject)
                            
                    ##these are tracks missed either because they disappeared 
                    ## or because they are temporarily invisable 
                    for index,obj in enumerate(available_tracks):
                        if index not in indexes_np[:,0]:
                            ## not update in this frame, increase frames_since_seen
                            obj.frames_since_seen += 1
                            ##but we update KF with predicted location
                            obj.kalman_update_missing(obj.predicted_position[-1])
                    
                    ##remove movingObj not updated for more than threasholds numbers of frames  
                    for index, obj in enumerate(current_tracks):
                        ##if a moving object hasn't been updated for 10 frames then remove it
                        h,w = postprocessed.shape[:2]
                        if obj.frames_since_seen > MISSING_THREASHOLD:
                            if(obj.position[-1][0] < BOUNDRY or obj.position[-1][0] > w-BOUNDRY or obj.position[-1][1] < BOUNDRY or obj.position[-1][1] > h-BOUNDRY):
                                print("Delete tracking", obj.position[-1][0], obj.position[-1][1])
                                del current_tracks[index]
                        elif (obj.frames_since_seen > MISSING_THREASHOLD_MAX):
                            print("Delete tracking over max missing threshold")
                            del current_tracks[index]
                        ## if the object is out of the scene then remove from current tracking right away
                        
                        #if (obj.position[-1][0] < 0 or obj.position[-1][0] > w):
                            #del current_tracks[index]

                        #elif (obj.position[-1][1] < 0 or obj.position[-1][1] > h):
                            #del current_tracks[index]
                            
                h,w = postprocessed.shape[:2]
                counter = "Pedestrians: " +  str(COUNTER_p) + " Cyclists: "+ str(COUNTER_c) + " frame: " + str(elapsed)
                cv2.putText(postprocessed,counter,(20,int(h-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
                #if(SAVE_FRAME):
                    #savepath = "./images/"
                    #cv2.imwrite(savepath, postprocessed)
                
                #########################################
                if SaveVideo:
                    videoWriter.write(postprocessed)
                if file == 0: #camera window
                    cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if file == 0: #camera window
            choice = cv2.waitKey(1)
            if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    if file == 0: #camera window
        cv2.destroyAllWindows()
        
    ##print out final report
    count = "Pedestrians: " +  str(COUNTER_p) + " Cyclists: "+ str(COUNTER_c)
    print(count)
    

def to_darknet(self):
    darknet_ckpt = self.darknet

    with self.graph.as_default() as g:
        for var in tf.global_variables():
            name = var.name.split(':')[0]
            var_name = name.split('-')
            l_idx = int(var_name[0])
            w_sig = var_name[1].split('/')[-1]
            l = darknet_ckpt.layers[l_idx]
            l.w[w_sig] = var.eval(self.sess)

    for layer in darknet_ckpt.layers:
        for ph in layer.h:
            layer.h[ph] = None

    return darknet_ckpt
