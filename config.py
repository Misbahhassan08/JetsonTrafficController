import os
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

usbCam = '/dev/video0'
video = '{}/testVideo2.mp4'.format(ROOT_DIR)
model = "ssd-mobilenet-v2"
model_th = 0.3
smallFont = 15
largeFont = 25
tracker_thresh = 50
conf_dist = 1/30
crop_factor = 1 #0.5
offset = 10
tempFolderPath = './temp'
speed_range = [20,60]
logging.basicConfig(filename="LogFile.log", level=logging.INFO)

show_outPut = False