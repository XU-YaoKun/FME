# NAME: loader.py
# DESCRIPTION: data loader for raw kitti data

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np 
np.set_printoptions(precision=4, suppress=True)
import scipy.misc
import pickle
import cv2
from glob import glob

from path import Path
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from torch.utils.data import Dataset
from .geom import get_episym 

# for test
from config import get_config

config, unparsed = get_config()

class CorrespondenceSet(Dataset):
    """
    dataset for raw kitti data
    [image, camera pose]
    """

    def __init__(self, 
                 dataset_dir,
                 nfeature=config.num_kp,
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
        
        # dataloader output
        self.ts = []
        self.Rs = []
        self.xs = []
        self.ys = []

        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.nfeature = nfeature

        self.cam_ids = ['02']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        self.get_X = get_X
        self.get_pose = get_pose
        

        if static_frames_file is not None:
            static_frames_file = Path(static_frames_file)
            self.collect_static_frames(static_frames_file)
        
        # get all drive path
        self.collect_train_folders()
        self.collect_drives()

        # get correspondenve and camera calibration matrix
        self.get_correspondence()
        
        xs_out = open("xs.pickle", "wb")
        pickle.dump(self.xs, xs_out)
        xs_out.close()
        
        ys_out = open("ys.pickle", "wb")
        pickle.dump(self.xs, ys_out)
        ys_out.close()

        Rs_out = open("Rs.pickle", "wb")
        pickle.dump(self.Rs, Rs_out)
        Rs_out.close()

        ts_out = open("ts.pickle", "wb")
        pickle.dump(self.ts, ts_out)
        ts_out.close()
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        out_dict = {}

        out_dict["Rs"] = self.Rs[index]
        out_dict["ts"] = self.ts[index]
        out_dict["xs"] = self.xs[index]
        out_dict["ys"] = self.ys[index]

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
        Find all in a drive directory
        """
        
        # init sift
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeature, contrastThreshold=1e-5)

        count = 0;
        index = 0;

        pair = []
        # al = len(self.d_scenes)
        al = 22382 

        scale = None

        for it1, it2 in zip(self.d_scenes[0::2],self.d_scenes[1::2]):
                print("\rLoad {} images out of {}".format(count*2, al), end="")
                drive1 = it1[:26]
                drive2 = it2[:26]
                if(drive1 != drive2): continue
                
                path1 = self.get_img_path(it1)
                path2 = self.get_img_path(it2)
                
                img1 = cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2GRAY)

                # height, width = img1.shape
                # print("height is {}, width is {}".format(height, width))

                kp1, des1 = sift.detectAndCompute(img1, None)
                kp2, des2 = sift.detectAndCompute(img2, None)
                xy1 = np.array([_kp.pt for _kp in kp1])
                xy2 = np.array([_kp.pt for _kp in kp2])
    
                # find correspondence according to description
                x1, x2 = self.knn_match(xy1, xy2, des1, des2, if_BF=True)
                # print("#"*25, " x1 ", "#"*25)
                # print(x1)
                # print("#"*25, " x2 ", "#"*25)
                # print(x2)
                
                xs = np.concatenate((x1, x2), axis=1)

                K, imu2rect = self.get_camera_pose(drive1)
                # print("camera intrinsic matrix")
                # print(K)

                oxt_path1 = self.get_oxt_path(it1) 
                oxt_path2 = self.get_oxt_path(it2)
                metadata1 = np.genfromtxt(oxt_path1)
                metadata2 = np.genfromtxt(oxt_path2)
                lat1 = metadata1[0]
                
                if scale is None:
                    scale = np.cos(lat1 * np.pi / 180.)

                imu_pose1 = self.get_imupos(metadata1[:6], scale)
                imu_pose2 = self.get_imupos(metadata2[:6], scale)
                
                odo_pose = imu2rect @ np.linalg.inv(imu_pose1) @ imu_pose2 @ np.linalg.inv(imu2rect)
                odo_pose_inv = np.linalg.inv(odo_pose)
                
                Rt = odo_pose

                rotation = Rt[:3, :3]
                translation = Rt[:3, 3:4]
                
                # print("odo_pose")
                # print(rotation)
                # print(translation)
                # calculate ys
                geod_d = get_episym(x1, x2, rotation, translation, K)
                ys = geod_d
                # print("\nnum of good cor {}\n".format(np.count_nonzero(ys<1e-4)))

                # print(des1)
                self.Rs.append(rotation)
                self.ts.append(translation)
                self.xs.append(xs)
                self.ys.append(ys)
                # print("\n")
                # print(odo_pose_inv)
                count = count + 1
    
    def knn_match(self, x1_all, x2_all, des1, des2, if_BF=False):
        """
        compute correspondence according to knn algorithm
        """

        if if_BF:
            bf = cv2.BFMatcher(normType=cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=1)

        # good = []
        # all_m = []

        # for m,n in matches:
        #     all_m.append(m)
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)
        
        x1 = x1_all[[mat[0].queryIdx for mat in matches], :]
        x2 = x2_all[[mat[0].trainIdx for mat in matches], :]
        # print("\n")
        # print("shape of x1: {}".format(x1.shape))
        # print("shape of x2: {}".format(x2.shape))
        assert x1.shape == x2.shape

        # print("# good points: {}".format(len(good)))

        return x1, x2
    
    def to_homo(self, R, T):
        """
        convert R(1x9),T(1x3) to homogeneous corrdinate Trans[4x4]
        """
        R = R.reshape(3,3)
        T = T.reshape(3,1)
        return np.vstack((np.hstack([R, T]), [0,0,0,1]))

    def get_camera_pose(self, drive_path):
        base_path = self.dataset_dir + drive_path[:10]
        
        imu2velo_dict = self.read_calib_file(base_path+'/calib_imu_to_velo.txt')
        velo2cam_dict = self.read_calib_file(base_path+'/calib_velo_to_cam.txt')
        cam2cam_dict = self.read_calib_file(base_path+'/calib_cam_to_cam.txt')

        imu2velo_mat = self.to_homo(imu2velo_dict["R"], imu2velo_dict["T"])
        velo2cam_mat = self.to_homo(velo2cam_dict['R'], velo2cam_dict['T'])
        cam2rect_mat = self.to_homo(cam2cam_dict['R_rect_00'], np.zeros(3))
       
        P_rect_20 = np.reshape(cam2cam_dict["P_rect_02"],(3,4))
        K = P_rect_20[:3, :3]
        Ml_gt = np.matmul(np.linalg.inv(K), P_rect_20)
        
        Rl_gt = Ml_gt[:, :3]
        tl_gt = Ml_gt[:, 3:4]
        Rtl_gt = np.vstack((np.hstack((Rl_gt, tl_gt)), np.array([0., 0., 0., 1.], dtype=np.float64)))
        
        # from imu coordinate to rectified camera coordinate
        imu2rect = Rtl_gt @ cam2rect_mat @ velo2cam_mat @ imu2velo_mat
        
        return K, imu2rect


    def read_calib_file(self, path):
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        pass
        return data
    
    def get_img_path(self, it):
        date = it[:10]
        drive = it[:26]
        camera = 'image_' + it[27:29]
        image_index = it[30:40]
        img_path = os.path.join(self.dataset_dir, date, drive, camera, 'data', image_index+'.png')
        return img_path

    def get_oxt_path(self, it):
        date = it[:10]
        drive = it[:26]
        image_index = it[30:40]
        oxt_path = os.path.join(self.dataset_dir, date, drive, 'oxts', 'data', image_index+'.txt')
        return oxt_path

    def get_imupos(self, metadata, scale):
        lat, lon, alt, roll, pitch, yaw = metadata
        er = 6378137. # earth radius (approx.) in meters
        ty = lat * np.pi * er / 180.
        tx = scale * lon * np.pi * er / 180.
        tz = alt
        t = np.array([tx, ty, tz]).reshape(-1,1)
        Rx = self.rotx(roll)
        Ry = self.roty(pitch)
        Rz = self.rotz(yaw)
        # print(Rz)
        R = Rz @ Ry @ Rx
        return self.to_homo(R, t)

    def rotx(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def rotz(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])



if __name__ == "__main__":
    dataset = CorrespondenceSet(config.dataset_dir,
                                1000,
                                config.static_frames_file,
                                config.test_scene_file,
                                img_height=config.img_height,
                                img_width=config.img_width)


