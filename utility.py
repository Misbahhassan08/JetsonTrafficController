import math


class object:
    def __init__(self):
        self.class_id = -1
        self.sy = 0
        self.ey = 0
        self.box = (0, 0, 0, 0)
        self.cnt_disp = 0
        self.cnt_life = 1
        self.is_updated = True
        self.speed = 0
        self.stime = None
        self.etime = None
        self.image = None
        self.is_uploaded = False


    def get_distance_(self):
        return math.hypot(self.scp[0] - self.ecp[0], self.scp[1] - self.ecp[1])

def get_center(left, top, right, bottom):
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    return [cx, cy]


def get_iou(bb1, bb2):

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area + 1e-10)
    return iou


