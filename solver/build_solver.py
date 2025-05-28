import torch
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import _LRScheduler

import torch

def build_optimizer(cfg, model):
    lr = cfg.SOLVER.LR
    decay = cfg.SOLVER.DECAY
    momentum = cfg.SOLVER.SGD_MOMENTUM

    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    elif cfg.SOLVER.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=decay)

    elif cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    else:
        raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER}")

    return optimizer

def build_scheduler(cfg, optimizer):
    total_epoch = cfg.SOLVER.EPOCH
    milestones = [9999999999999]
    gamma = cfg.SOLVER.STEP_GAMMA

    if cfg.SOLVER.SCHEDULER == "Linear":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {cfg.SOLVER.LR_SCHEDULE}")
    return scheduler

