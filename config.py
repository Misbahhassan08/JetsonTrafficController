import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


video = '{}/testVideo2.mp4'.format(ROOT_DIR)
model = "ssd-mobilenet-v2"
model_th = 0.7