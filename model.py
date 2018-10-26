import tensorflow as tf
from resnet import Resnet
from rpn import RPN
from pyramid_roi_pooling import PyramidRoiPooling


class MRCNN:
    def __init__(self, roi_size):
        self.roi_size = roi_size
        self.pyramidRoiPooling = PyramidRoiPooling(roi_size)
