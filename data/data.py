# NAME: data.py
# Description: load data from kitti dataset and use SIFT to extract key point

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pickle

import h5py
import numpy as np 
import cv2

from config import get_config
from loader import KittiLoader

# get argument
config, unparsed = get_config()

data_loader = KittiLoader(config.dataset_dir,
                          config.static_frames_file,
                          config.test_scene_file,
                          img_height=config.img_height,
                          img_width=config.img_width)





