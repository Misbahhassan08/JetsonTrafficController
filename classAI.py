import cv2
import jetson.inference
import jetson.utils
import time
import threading
from PIL import Image
import numpy as np
import os
import cv2
from config import ROOT_DIR, video, model, model_th
from utility import *

class AI(threading.Thread):
    def __init__(self, conf):
        threading.Thread.__init__(self)
        self.ROOT_DIR = ROOT_DIR
        self.net = jetson.inference.detectNet(model, model_th)
        self.video = video
        self.display = jetson.utils.videoOutput()
        self.config_ini = conf
        #camera = jetson.utils.videoSource(config_ini['CAMERA']['rtspurl'],argv=['--input-codec=h264','--input-rtsp-latency=0'])
        self.camera = jetson.utils.videoSource(self.video)

        self.show_line = 550
        self.dist_conf = 18 / 300  # 18 meters / 300 pixels

        self.obj_list = []
        self.frame_data = None
        self.thresh_speed = 10, 100  # k/hr
        self.vimage_size = 120, 120
        self.basewidth = 200
        self.font_size = 15.0
        self.cur_failed = 0
        self.cnt_saved = 0
        self.is_running = True

        self.speed_text = 'Approx speeds :'
        pass # end of __init__ function


    def run(self):
        
        #while True:
        while self.display.IsStreaming():
            max_failed = 20
            try:
                img = self.camera.Capture()
                self.cur_failed = 0
            except:
                print("Disconnected....")
                self.cur_failed += 1
                if self.cur_failed > max_failed:
                    self.is_running = False

            if img is not None:
                try:
                    frame_data = [img, time.time()]
                    # frame_data.append([img, time.time()])
                except:
                    pass

            cuimg, dtime = frame_data.copy()
            # backup cuda-image
            cuimg_copy = jetson.utils.cudaAllocMapped(width=cuimg.width, height=cuimg.height, format=cuimg.format)

            # copy the image (dst, src)
            jetson.utils.cudaMemcpy(cuimg_copy, cuimg)

            detections = self.net.Detect(cuimg)
            removable_boxes = []
            updatable_objes = []
            addable_boxes = []
            is_show = False
            for detection in detections:
                if detection.Bottom > self.show_line:
                    is_show = True

                # find matched object
                max_iou = 0.0
                obj_id = -1
                for i, obj in enumerate(self.obj_list):
                    if obj.class_id != detection.ClassID:
                        continue
                    iou = get_iou(obj.box, [detection.Left, detection.Top, detection.Right, detection.Bottom])
                    if iou > 0.05 and iou < 1.0:
                        max_iou = iou
                        obj_id = i

                if obj_id < 0:
                    nobj = object()
                    nobj.class_id = detection.ClassID
                    nobj.sy = nobj.ey = detection.Bottom
                    nobj.box = (detection.Left, detection.Top, detection.Right, detection.Bottom)


                    # crop the image to the ROI
                    crop_roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
                    imgOutput = jetson.utils.cudaAllocMapped(width=crop_roi[2] - crop_roi[0],
                                                            height=crop_roi[3] - crop_roi[1],
                                                            format=cuimg_copy.format)
                    jetson.utils.cudaCrop(cuimg_copy, imgOutput, crop_roi)

                    nobj.cuimage = imgOutput
                    nobj.stime = nobj.etime = dtime
                    self.obj_list.append(nobj)

                else:
                    if detection.Bottom < self.show_line:
                        self.obj_list[obj_id].etime = dtime
                        self.obj_list[obj_id].ey = detection.Bottom
                        self.obj_list[obj_id].cnt_life += 1

                    if self.obj_list[obj_id].speed < 1:
                        crop_roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
                        imgOutput = jetson.utils.cudaAllocMapped(width=crop_roi[2] - crop_roi[0],
                                                                height=crop_roi[3] - crop_roi[1],
                                                                format=cuimg_copy.format)
                        jetson.utils.cudaCrop(cuimg_copy, imgOutput, crop_roi)
                        self.obj_list[obj_id].image = imgOutput

                    self.obj_list[obj_id].box = (detection.Left, detection.Top, detection.Right, detection.Bottom)
                    self.obj_list[obj_id].is_updated = True
                    self.obj_list[obj_id].cnt_disp = 0

            for id, obj in enumerate(self.obj_list):
                if not obj.is_updated:
                    self.obj_list[id].cnt_disp += 1
                self.obj_list[id].is_updated = False

            obj_list = [obj for obj in self.obj_list if obj.cnt_disp < 4]

            show_objs = []

            if is_show:
                for id, obj in enumerate(obj_list):
                    # calculate speed
                    if obj.speed < 1 and obj.box[3] > self.show_line and obj.cnt_life > 2:
                        dist_pixes = obj.ey - obj.sy
                        meters = dist_pixes * self.dist_conf
                        est = obj.etime - obj.stime

                        obj_list[id].speed = 3.6 * meters / est

                    # if obj_list[id].speed > 1 and obj_list[id].is_updated:
                    if obj_list[id].speed > self.thresh_speed[0] and obj_list[id].speed < self.thresh_speed[1]:
                        show_objs.append([obj_list[id].speed, obj_list[id].box[0]])

                        if not obj_list[id].is_uploaded:
                            name = f"{str(int(time.time() * 100000))[5:]}.png"

                            cuimage = obj_list[id].cuimage
                            _box = obj_list[id].box

                            print('************************************* ',_box)
                            font = jetson.utils.cudaFont(size=self.font_size)
                            font2 = jetson.utils.cudaFont(size=25)

                            font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Camera ID : {self.config_ini['CAMERA']['cameraid']}", 5, 15, font.Yellow, font.Gray40)
                            font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Date : {time.strftime('%m-%d-%Y')}", 5, 27, font.Yellow, font.Gray40)
                            font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Time : {time.strftime('%H:%M:%S')}", 5, 40, font.Yellow, font.Gray40)
                            font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Speed : {int(obj_list[id].speed)}", 5, 55, font.Yellow, font.Gray40)
                            font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Site Name : {self.config_ini['CAMERA']['sitename']}", 5, 70, font.Yellow, font.Gray40)


                            font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"Camera ID : {self.config_ini['CAMERA']['cameraid']}", int(_box[0]), int(_box[1]), font.Yellow, font.Gray40)
                            font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"Date : {time.strftime('%m-%d-%Y')}", int(_box[0]),int(_box[1])+15, font.Yellow, font.Gray40)
                            font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"Time : {time.strftime('%H:%M:%S')}", int(_box[0]),int(_box[1])+30, font.Yellow, font.Gray40)
                            font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"Speed : {int(obj_list[id].speed)}", int(_box[0]),int(_box[1])+45, font.Yellow, font.Gray40)
                            font2.OverlayText(cuimg, cuimg.width, cuimg.height, f"Site Name : {self.config_ini['CAMERA']['sitename']}", int(_box[0]),int(_box[1])+60, font.Yellow, font.Gray40)

                            jetson.utils.saveImage(os.path.join('./temp', name), cuimage)
                            self.cnt_saved += 1
                            print("saved : ", self.cnt_saved)
                            obj_list[id].is_uploaded = True

                if show_objs.__len__() > 0:
                    speeds = sorted(show_objs, key=lambda a:a[1])

                    self.speed_text = 'Approx speeds :'
                    for speed, _ in speeds:
                        self.speed_text += " {:.0f} km/hr, ".format(speed)
                    self.speed_text = self.speed_text[:-2]
                    # print("speed : ", speed_text)

                    # display.SetStatus("approx speed in " + state_text)
            
            # convert to BGR, since that's what OpenCV expects
            bgr_img = jetson.utils.cudaAllocMapped(width=cuimg.width,
                                            height=cuimg.height,
                                            format='bgr8')

            jetson.utils.cudaConvertColor(cuimg, bgr_img)

            # make sure the GPU is done work before we convert to cv2
            jetson.utils.cudaDeviceSynchronize()

            # convert to cv2 image (cv2 images are numpy arrays)
            cv_img = jetson.utils.cudaToNumpy(bgr_img)
            cv2.line(cv_img,(0,self.show_line),(1780,self.show_line),(0,0,255),1)
            cv2.putText(cv_img, "Processing {:.0f} FPS {}{}".format(self.net.GetNetworkFPS(), " "*50, self.speed_text), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("window", cv_img)
            cv2.waitKey(1)
            #self.display.SetStatus("Processing {:.0f} FPS {}{}".format(self.net.GetNetworkFPS(), " "*50, self.speed_text))
            #self.display.Render(img)

        
        pass # end of run function

