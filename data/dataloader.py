# NAME: dataloader.py
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

import scipy.misc
from config import get_config
from .dataset import CorrespondenceSet 
from joblib import Parallel, delayed

from torch.utils.data import DataLoader

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_cor(n, args):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    correspondence = data_loader.get_cor(n)
    f1, f2 = data_loader.get_f(n)

# get argument
config, unparsed = get_config()

print(config.test_scene_file)

dataset = CorrespondenceSet(config.dataset_dir,
                            config.num_kp,
                            config.static_frames_file,
                            config.test_scene_file,
                            img_height=config.img_height,
                            img_width=config.img_width)

train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
print("\n")
# test_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

# for it in data_loader.drives:
#     print(it)



