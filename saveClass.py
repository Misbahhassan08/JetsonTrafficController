import cv2
import jetson.utils
import time
import threading
import os
from config import *

class Saving(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_running = True
        self.upload = False
        self.img = None
        self.config = None
        self.font = None
        self.speed = 0
        pass


    def getImage(self, Img, config, font, speed):
        self.img = Img
        self.config = config
        self.font = font
        self.speed = speed
        self.upload = True

    def stopThread(self):
        self.is_running = False
        self.join()

    def setImages(self):
        # self.car = id : img, _, _, speed, time
        if True:
            try:
                cuimage = self.img
                config_ini = self.config
                font = self.font
                font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Camera ID : {config_ini['CAMERA']['cameraid']}", 5, 15, font.Yellow, font.Gray40)
                font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Date : {time.strftime('%m-%d-%Y')}", 5, 25, font.Yellow, font.Gray40)
                font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Time : {time.strftime('%H:%M:%S')}", 5, 35, font.Yellow, font.Gray40)
                font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Speed : {int(self.speed)}", 5, 45, font.Yellow, font.Gray40)
                font.OverlayText(cuimage, cuimage.width, cuimage.height, f"Site Name : {config_ini['CAMERA']['sitename']}", 5, 55, font.Yellow, font.Gray40)
                bgr_img = jetson.utils.cudaAllocMapped(width=cuimage.width,
                                            height=cuimage.height,
                                            format='bgr8')

                jetson.utils.cudaConvertColor(cuimage, bgr_img)

                # make sure the GPU is done work before we convert to cv2
                jetson.utils.cudaDeviceSynchronize()
                cv_img = jetson.utils.cudaToNumpy(bgr_img)
                name = f"{str(int( time.time() * 100000))[5:]}.png"
                #print("######################### IMAGE ##################\n",cv_img)
                cv2.imwrite(os.path.join(tempFolderPath, name), cv_img)
         
            except Exception as error :
                print("FILE UPLOAD ERROR : ",error)
                self.logUpdater(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR IN SAVING Images: {error}")
                self.stopThread()
                self.health = False
    
    def getHealth(self):
        return self.health
    
    def logUpdater(self, mesg):
        logging.info('{}'.format(mesg))
    
    def run(self):
        while self.is_running:
            try:
                if self.upload is True:
                    self.setImages()
                    self.upload = False
            except Exception as error:
                self.logUpdater(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR IN SAVING Images: {error}")
                self.stopThread()

