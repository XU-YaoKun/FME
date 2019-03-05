# NAME: loader.py
# DESCRIPTION: data loader for raw kitti data

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np 
import scipy.misc
import os
import cv2
from glob import glob

from path import Path
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from torch.utils.data import Dataset

# for test
from config import get_config

config, unparsed = get_config()

class CorrespondenceSet(Dataset):
    """
    data loader for raw kitti data
    [image, camera pose]
    """

    def __init__(self, 
                 dataset_dir,
                 nfeature=1000,
                 static_frames_file=None,
                 test_scene_file=None,
                 img_height=128,
                 img_width=416,
                 get_X=False,
                 get_pose=False):
        dir_path = Path(__file__).realpath().dirname()

        # print(type(test_scene_file))
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()

        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.correspondence = []
        self.F = []
        self.description = []
        self.nfeature = nfeature

        self.cam_ids = ['02']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        self.get_X = get_X
        self.get_pose = get_pose
        

        if static_frames_file is not None:
            static_frames_file = Path(static_frames_file)
            self.collect_static_frames(static_frames_file)

        self.collect_train_folders()
        self.collect_drives()
        self.get_correspondence()

    def __len__(self):
        return len(self.correspondence)

    def __getitem__(self, index):
        out_dict = {}

        out_dict["correspondence"] = self.correspondence[index]
        out_dict["F"] = self.F[index]
        out_dict["description"] = self.description[index]

        return out_dict

    def collect_static_frames(self, static_frames_file):
        """
        Get all frames in static file, and remove them from data loader
        """
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()

        self.static_frames = []

        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % np.int(frame_id[:-1])
            for cid in self.cam_ids:
                self.static_frames.append(drive+' '+cid+' '+curr_fid)
        logging.info('Static frames has been collected from %s'%static_frames_file)

    
    def collect_train_folders(self):
        """
        Get all frames in train 
        """

        # dynamic scenes
        self.d_scenes = []
        for date in self.date_list:
            drive_set = os.listdir(self.dataset_dir+date+'/') 
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, date, dr)
                if os.path.isdir(drive_dir):
                    if dr[:-5] not in self.test_scenes:
                        for cam in self.cam_ids:
                            img_dir = os.path.join(drive_dir, 'image_'+cam, 'data')
                            N = len(glob(img_dir+'/*.png'))
                            for n in range(N):
                                frame_id = '%.10d'%n
                                self.d_scenes.append(dr+' '+cam+' '+frame_id)
        
        # remove static frame from train data
        for s in self.static_frames:
            try:
                self.d_scenes.remove(s)
            except:
                pass
        

    def collect_drives(self):
        self.drives = []
        for date in self.date_list:
            drive_set = (self.dataset_dir/date).dirs()
            for dr in drive_set:
                self.drives.append(dr)
        self.num_train = len(self.drives)

    
    def get_correspondence(self):
        """
        Find all correspondence in a drive directory
        """
        
        # init sift
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeature, contrastThreshold=1e-5)

        count = 0;
        index = 0;

        pair = []
        al = len(self.d_scenes)
        for it1, it2 in zip(self.d_scenes[0::2],self.d_scenes[1::2]):
                print("\rLoad {} images out of {}".format(count*2, al), end="")
                drive1 = it1[:26]
                drive2 = it2[:26]
                if(drive1 != drive2): continue
 
                path1 = self.get_path(it1)
                path2 = self.get_path(it2)
                # print(path1)
                # print(path2)
                img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
                img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
                count = count + 1
    
    def get_path(self, it):
        date = it[:10]
        drive = it[:26]
        camera = 'image_' + it[27:29]
        image_index = it[30:40]
        img_path = os.path.join(self.dataset_dir, date, drive, camera, 'data', image_index+'.png')
        return img_path

dataset = CorrespondenceSet(config.dataset_dir,
                            1000,
                            config.static_frames_file,
                            config.test_scene_file,
                            img_height=config.img_height,
                            img_width=config.img_width)


# for it in dataset.d_scenes:
        # print(it)
