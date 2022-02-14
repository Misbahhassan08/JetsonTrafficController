import sys
sys.path.insert(0, './sort')

import jetson.inference
import jetson.utils
import time
import threading
import numpy as np
import os
import cv2
from config import *
import random
from saveClass import Saving
from sort import *

class AI(threading.Thread):
    def __init__(self, config):
        threading.Thread.__init__(self)
        if not os.path.isdir(tempFolderPath):
            os.mkdir(tempFolderPath)
        self.config = config
        self.sav = Saving()
        self.sav.start()
        self.ROOT_DIR = ROOT_DIR
        self.net = jetson.inference.detectNet(model, model_th)
        self.video = video
        #self.display = jetson.utils.videoOutput()
        #camera = jetson.utils.videoSource(config_ini['CAMERA']['rtspurl'],argv=['--input-codec=h264','--input-rtsp-latency=0'])
        self.camera = jetson.utils.videoSource(video)

        self.font_size1 = smallFont 
        self.font_size2 = largeFont
        self.is_running = True
        self.sort_tracker = Sort(max_age=5,
                                min_hits=2,
                                iou_threshold=0.2) 
        self.vehc = {}
        self.last_id_deleted = 0

        pass # end of __init__ function

    def closeProject(self):
        self.is_running = False
        self.sav.stopThread()
        self.is_running = False
        self.join


    def logUpdater(self, mesg):
        logging.info('{}'.format(mesg))

    def cropImage(self,img,  bbox):
        x1, y1, x2, y2 = bbox
        left = x1 + offset
        top = y1 + offset
        right = x2 + offset
        bottom = y2 + offset

        

        if left < 0: left = 0
        if right < 0: right = 0
        if top < 0: top = 0
        if bottom < 0: bottom = 0

        if right > img.width: right = img.width
        if bottom > img.height: bottom = img.height

        

        roi_width = right - left
        roi_height = bottom - top
        crop_roi = (left, top, right, bottom)   
        #self.logUpdater(f"ORIGNAL IMAGE width:{img.width} : height:{img.height}")
        #self.logUpdater(f"ROI : left:{left} top:{top} right:{right} bottom:{bottom} width:{roi_width} height:{roi_height }")

        imgOutput = jetson.utils.cudaAllocMapped(width=roi_width,
                                                height=roi_height,
                                                format=img.format)
        jetson.utils.cudaCrop(img, imgOutput, crop_roi)
        return imgOutput



    def run(self):
        self.logUpdater('########################   Program Starting  : {} #######################'.format(time.time()))
        #while True:
        try:
            font = jetson.utils.cudaFont(size=self.font_size1)
            font2 = jetson.utils.cudaFont(size=self.font_size2)
            while self.is_running:
                self.logUpdater('............................ ..................NEXT LOOP .........................')
                # reset lists 

                img = self.camera.Capture()
                frame_data = [img, time.time()]
                cuimg, dtime = frame_data
                t = dtime 
                detections = self.net.Detect(cuimg, overlay=",,conf" )
                dets_to_sort = np.empty((0,6))

                for detection in detections:
                    detclass, conf ,x, y, w, h, b, r =detection.ClassID, detection.Confidence, int(detection.Left), int(detection.Top), int(detection.Width), int(detection.Height),  int(detection.Bottom),  int(detection.Right)
                    dets_to_sort = np.vstack((dets_to_sort, np.array([x, y, r, b, conf, detclass])))
                    
                
                tracked_dets = self.sort_tracker.update(dets_to_sort)
                
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    for i ,box in enumerate(bbox_xyxy):
                        
                        x1, y1, x2, y2 = [int(i) for i in box]
                        # box text and bar
                        cat = int(categories[i]) if categories is not None else 0
                        id = int(identities[i]) if identities is not None else 0
                        
                        img = self.cropImage(cuimg, (x1, y1, x2, y2))
                        try:
                            self.vehc[id][1] = abs(t - self.vehc[id][1])
                            self.vehc[id][2] = abs(y1 - self.vehc[id][2])
                            meters =  self.vehc[id][2] * conf_dist
                            speed =  int(3.6* meters/self.vehc[id][1])
                            if speed < speed_range[0]: speed = (random.randrange(speed_range[0],speed_range[0]+20)) 
                            if speed > speed_range[1]: speed = (random.randrange(speed_range[1],speed_range[1]+20)) 
                            self.vehc[id][3] = speed
                            self.vehc[id][0] = img
                            self.vehc[id][4] = t

                            self.sav.getImage(img, self.config, font, speed)
                        except Exception as error:
                            self.logUpdater(f"########################################### WARN: {error} ###########")
                            self.vehc[id] = [img, t ,y1, speed_range[0], t] # box , time difference , pixels difference, speed, orignal time
                            pass
                        self.logUpdater(f"*********** FPS ******** {self.net.GetNetworkFPS()}")
                        #self.logUpdater(f"Car ID:{id}, Cat:{cat}, box:{x1,y1,x2,y2} pixelDiff:{self.vehc[id][2]} timeDiff:{self.vehc[id][1]} Speed:{self.vehc[id][3]}")
                        #self.logUpdater(f"UPDATED VEH LIST : {self.vehc}")
                        if show_outPut:
                            font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"ID :{id} SPEED:{self.vehc[id][3]}", x1, y1+40, font.Yellow, font.Gray40)
                            jetson.utils.cudaDrawRect(cuimg, (x1,y1,x2,y2), (25,127,89,50))
                        if len(self.vehc) > 20:
                            # delete first 10 car id's 
                            
                            for x in range(10):
                                self.last_id_deleted  += 1
                                try:
                                    del self.vehc[self.last_id_deleted]
                                except:
                                    pass
                        pass
                #self.net.PrintProfilerTimes()
                if show_outPut:
                    font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"FPS : {self.net.GetNetworkFPS()}", 5, 30, font.Yellow, font.Gray40)
                    bgr_img = jetson.utils.cudaAllocMapped(width=cuimg.width,
                                            height=cuimg.height,
                                            format='bgr8')

                    jetson.utils.cudaConvertColor(cuimg, bgr_img)

                    # make sure the GPU is done work before we convert to cv2
                    jetson.utils.cudaDeviceSynchronize()
                    cv_img = jetson.utils.cudaToNumpy(bgr_img)
                    cv2.imshow("Window", cv_img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        self.closeProject()
                        break
        except Exception as error:
            print("ERROR in AI LOOP : ",error)
            self.logUpdater(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR IN MAIN : {error}")
            self.closeProject()
        pass # end of run function

#ai = AI()
#ai.run()