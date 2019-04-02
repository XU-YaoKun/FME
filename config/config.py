# NAME: config.py 
# DESCRIPTION: Based on argparse usage

import argparse

arg_list = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument("--net_depth", type=int, default=12, help="number of layers")
net_arg.add_argument("--net_channel", type=int, default=128, help="number of channels in each layer")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument("--dataset_dir", type=str, default="/data/KITTI/raw_meta/", help="path to dataset")
data_arg.add_argument("--num_threads", type=int, default=6, help="number of thread to load data")
data_arg.add_argument("--img_height", type=int, default=128, help="number of thread to load data")
data_arg.add_argument("--img_width", type=int, default=418, help="number of thread to load data")
data_arg.add_argument("--static_frames_file", type=str, default="ref/static_frames.txt", help="static data file path")
data_arg.add_argument("--test_scene_file", type=str, default="ref/test_scenes_eigen.txt", help="test data file path")
data_arg.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")

# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("Object")
obj_arg.add_argument("--num_kp", type=int, default=2000, help="number of keypoints per image")
obj_arg.add_argument("--top_k", type=int, default=2000, help="number of keypoint to use for estimation")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("Loss")
loss_arg.add_argument("--classif_weight", type=float, default=1.0, help="weight for classification loss")
loss_arg.add_argument("--essential_weight", type=float, default=0.1, help="weight for essential matrix loss")
loss_arg.add_argument("--threshold", type=float, default=1e-4, help="threshold for determining good correspondence")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--max_epoch", type=int, default=200, help="max epoch to train the model")
train_arg.add_argument("--batch_size", type=int, default=32, help="batch size")
train_arg.add_argument("--output_dir", type=str, default="../output", help="output file for training log")
train_arg.add_argument("--lr", type=float, default=1e-3, help="learning rate for training")
train_arg.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for training")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

#
# config.py ens here
