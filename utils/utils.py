import argparse
import torch
import os
import sys
import logging
from torchvision import transforms
import functools
import shutil
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode
import importlib.util


def load_module_from_path(file_path, module_name='dynamic_module'):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def SetEverything(config, rank):
    parent_path = config.OUTPUT_DIR + "/" + config.EXP_NAME
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join(parent_path, time)

    tb_path = check_dir(out_dir + config.TB)  # tensorboard
    writer = SummaryWriter(tb_path)

    ckpt_path = check_dir(out_dir + config.CKPT)  # checkpoint
    log_path = check_dir(out_dir + config.LOG)  # log

    check_file(out_dir, "config.yml")
    save_cfg_to_yaml(config, os.path.join(out_dir, "config.yml"))  # have bugs, I save it just for self-check

    logger = setup_logger(config.EXP_NAME,
                          log_path,
                          rank,
                          "log.txt")
    return logger, writer, ckpt_path, out_dir


def cfg_node_to_dict(cfg_node):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    cfg_dict = {}
    for key, value in cfg_node.items():
        if isinstance(value, CfgNode):
            cfg_dict[key] = cfg_node_to_dict(value)
        else:
            cfg_dict[key] = value
    return cfg_dict


def save_cfg_to_yaml(cfg, filepath):
    cfg_dict = cfg_node_to_dict(cfg)
    with open(filepath, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def check_dir(dir):
    # use try except, instead if not path.exists
    # as it may meet files exist error when multi-GPUs
    # a_process -> os.mkdir
    # os.mkdir -> b_process
    try:
        os.makedirs(dir)
    except OSError:
        pass
    return dir


def check_file(dir, filename):
    if os.path.exists(os.path.join(dir, filename)):
        pass
    else:
        with open(os.path.join(dir, filename), "w") as file:
            print("{file} created successfully.".format(file=filename))


def setup_logger(name, save_dir, distributed_rank, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def reverse_normalize(normalized_image):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    inv_tensor = inv_normalize(normalized_image)
    return inv_tensor


def Tensor2PIL(img_tensor):
    func = transforms.ToPILImage()
    return func(img_tensor)


def copy_folder_without_images(src_folder, dest_folder, image_extensions=('.jpg', '.jpeg', '.png', '.gif')):
    # copy the folder name for save results, avoiding process preemption in acceleration when mkdir
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Walk through the source folder
    for root, dirs, files in os.walk(src_folder):
        # Exclude images with specified extensions
        files = [file for file in files if not file.endswith(image_extensions)]

        # Create the corresponding sub-directory structure in the destination folder
        dest_root = os.path.join(dest_folder, os.path.relpath(root, src_folder))
        if not os.path.exists(dest_root):
            os.makedirs(dest_root)

        # Copy non-image files to the destination
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            shutil.copy2(src_file, dest_file)  # Use shutil.copy2 to preserve file metadata


def cal_param(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    return Total_params/1e6, Trainable_params/1e6, NonTrainable_params/1e6


def l2_norm(hidden_states):
    return hidden_states.norm(p=2, dim=-1, keepdim=True)
