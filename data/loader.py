# NAME: loader.py
# DESCRIPTION: data loader for raw kitti data

import numpy as np 
import scipy.misc
import os
from glob import glob

from path import Path
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class KittiLoader():
    """
    data loader for raw kitti data
    [image, camera pose]
    """

    def __init__(self, 
                 dataset_dir,
                 static_frames_file=None,
                 test_scene_file=None,
                 img_height=128,
                 img_width=416,
                 get_X=False,
                 get_pose=False):
        dir_path = Path(__file__).realpath().dirname()

        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()

        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width

        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        self.get_X = get_X
        self.get_pose = get_pose
        

        if static_frames_file is not None:
            static_frames_file = Path(static_frames_file)
            self.collect_static_frames(static_frames_file)

        self.collect_train_folders()

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
        self.scenes = []
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
                                self.scenes.append(dr+' '+cam+' '+frame_id)
        
        # remove static frame from train data
        for s in self.static_frames:
            try:
                print(s)
                self.scenes.remove(s)
            except:
                pass
        
        self.num_train = len(self.scenes)

    def get_drive_path(self, date, drive):
        drive_path = self.dataset_dir + '/%s/%s_drive_%s_sync'%(date, date, drive)
        return drive_path

    def load_image_raw(self, scenes):
        img = scipy.misc.imread(scenes)
        return img


