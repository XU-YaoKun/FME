# Name: load.py
# Description: load data from pickle file and creat a dataloader given that data file

import os
from path import Path
import pickle 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

class PickleDataSet(Dataset):
    
    def __init__(self, pickle_path):
        super(PickleDataSet, self).__init__()
        self.xs = []
        self.ys = []
        self.Rs = []
        self.ts = []
        
        self.load_pickle(pickle_path)

    def load_pickle(self, pickle_path):
        xs_path = os.path.join(pickle_path, "xs.pickle")
        ys_path = os.path.join(pickle_path, "ys.pickle")
        Rs_path = os.path.join(pickle_path, "Rs.pickle")
        ts_path = os.path.join(pickle_path, "ts.pickle")

        xs_in = open(xs_path, 'rb')
        ys_in = open(ys_path, 'rb')
        Rs_in = open(Rs_path, 'rb')
        ts_in = open(ts_path, 'rb')

        self.xs = pickle.load(xs_in)
        self.ys = pickle.load(ys_in)
        self.Rs = pickle.load(Rs_in)
        self.ts = pickle.load(ts_in)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        out_dict = {}

        if(len(self.xs[index]) > 2000): 
            self.xs[index] = self.xs[index][0:2000]
            self.ys[index] = self.ys[index][0:2000]
            self.Rs[index] = self.Rs[index][0:2000]
            self.ts[index] = self.ts[index][0:2000]
        out_dict["x_in"] = self.xs[index]
        out_dict["y_in"] = self.ys[index]
        out_dict["R"] = self.Rs[index]
        out_dict["t"] = self.ts[index]

        return out_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
pickle_path = os.path.join(ROOT_DIR, "pickle")

dataset = PickleDataSet(pickle_path)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
