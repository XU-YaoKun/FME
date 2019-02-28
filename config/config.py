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


# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument("--data_path", type=str, default="../dataset", help="path to dataset")
data_arg.add_argument("--num_threads", type=int, default=6, help="number of thread to load data")

# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("Object")
obj_arg.add_argument("--num_kp", type=int, default=2000, help="number of keypoints per image")
obj_arg.add_argument("--top_k", type=int, default=2000, help="number of keypoint to use for estimation")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("Loss")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--batch_size", type=int, default=32, help="batch size")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

#
# config.py ens here