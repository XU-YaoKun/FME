# NAME: train.py
# DESCRIPTION: train models

import logging
import time

import torch
import torch.nn as nn

from fme.models.LGR.LGR import LGR
from fme.models.LGR.loss import LGR_loss
from fme.utils.checkpoint import Checkpointer
from fme.data.dataloader import train_dataloder, val_dataloader 

from config.config import get_config

config, unused = get_config()

def train_model():
    """
    train model in a single training epoch
    """

def eval_model():
    """
    evaluate model in a single validation epoch
    """

def train(config):
	"""
	train model
	"""
	output_dir = config.output_dir
	
	logger = logging.getLogger("fme.trainer")

    # Build model
	model = LGR()

    # Build loss
	loss = LGR_loss()

    # Build optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Build lr scheduler
	scheduler = torch.optim.lr_scheduler.MultiStepLR()

    # Build checkpointer
	checkpointer = Checkpointer(model, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir)
	
    # Build dataloader 
	train_dataloader = Dataloader(mode="train")  
	val_dataloader = Dataloader(mode="val")

    # train
    max_epoch = config.max_epoch
    

