### Dataset and Dataloader

------

I got correspondences and their descriptions from kitti using SIFT. And also, I extracted their relative pose from raw kitti dataset.

What I am concerned about is dynamic scene instead of static one. So I filter these types of images according to the static.txt, which I got from [SfMLearner](https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/static_frames.txt).

This dataloader is adapted from [Rui's repo](https://github.com/Jerrypiglet/kitti_instance_RGBD_utils/blob/master/KITTI_5_RANSAC_sample_twoFrame.ipynb). If you would like to get more details, check that one.



##### To do

If needed, I will dump the preprocessing data into pickle file, so that it will be much faster when training.

