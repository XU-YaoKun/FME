# NAME: train.py
# DESCRIPTION: train models

import logging
import time

import torch
import torch.nn as nn

from fme.models.LGR.LGR import LGR
from fme.models.LGR.loss import LGR_loss
from fme.utils.checkpoint import Checkpointer
# from fme.data.dataloader import train_dataloder, val_dataloader 
# from data.dataloader import train_dataloader

from data.pickle_loader import train_dataloader 
from config.config import get_config

config, unused = get_config()

def train_model(model, loss, dataloader, optimizer, log_period):
    """
    train model in a single training epoch
    """

    model.train()
    loss.train()

    end = time.time()
    for iteration, data_batch in enumerate(dataloader):
        data_time = time.time() - end
        data_batch = {k: v.cuda(non_blocking=True).float() for k, v in data_batch.items()}
        data_batch["x_in"] = data_batch["x_in"].transpose(1, 2).unsqueeze(2)
        data_batch["y_in"] = data_batch["y_in"].transpose(1, 2)

        batch_size = data_batch["x_in"].size(0)

        data_batch["R"] = data_batch["R"].view(batch_size, 9)
        data_batch["t"] = data_batch["t"].view(batch_size, 3)
        # print(data_batch["x_in"].size())
        # print(data_batch["y_in"].size())
        # print(data_batch["R"].size())
        # print(data_batch["t_in"].size())
        preds = model(data_batch)
        optimizer.zero_grad()
        loss_value = loss(data_batch, preds)
        loss_value.backward()
        optimizer.step()

        batch_time = time.time() - end
        
        if iteration % log_period == 0:
            print("iteration: {iter:4d}, lr: {lr:.2e}, loss: {loss:.2e}".format(iter=iteration, lr=optimizer.param_groups[0]["lr"], loss=loss_value))

    return loss_value



def eval_model():
    """
    evaluate model in a single validation epoch
    """

def train(config):
    """
    train model
    """
    output_dir = config.output_dir

    # Build model
    model = LGR(config)
    print("Build model: \n {}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # Build loss
    loss = LGR_loss(config.threshold, config.classif_weight, config.essential_weight)

    # Build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Build lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1)

    # Build checkpointer
    checkpointer = Checkpointer(model, optimizer=optimizer, scheduler=scheduler, save_dir=output_dir)
	
    # Build dataloader, now dataloader is created in data/dataloader.py 

    # train
    max_epoch = config.max_epoch

    for epoch in range(max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()

        loss_value = train_model(model, loss, train_dataloader, optimizer, log_period=config.log_period)

        epoch_time = time.time - start_time

        logger.info("Epoch[{}]-Train loss: {} Total_time: {:.2f}s".format(cur_epoch, loss_value, epoch_time))
    
        if cur_epoch == max_epoch:
            checkpointer.save("model_{:03d}".format(curepoch))

    return model

if __name__ == "__main__":
    train(config)
