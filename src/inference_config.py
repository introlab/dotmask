import os
import sys

sys.path.append('../nn/Mask_RCNN/')
from mrcnn import config

class InferenceConfig(config.Config):
    NAME = "ISM mrcnn"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80 