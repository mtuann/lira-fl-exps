import sys
import os
import pickle
import pathlib
import argparse


from torch import nn
import torch

import yaml
# from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np

# import seaborn as sns
# from tqdm.auto import tqdm
# from termcolor import colored

# from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import time

# from utils.dataloader import get_dataloader, PostTensorTransform, IMAGENET_MIN, IMAGENET_MAX
# from utils.backdoor import get_target_transform
# from utils.dnn import clear_grad

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()

loss_fn = nn.CrossEntropyLoss()

# def clip_image(x, dataset="cifar10"):
#     if dataset in ['tiny-imagenet', 'tiny-imagenet32']:
#         return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     elif args.dataset == 'cifar10':
#         return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     elif args.dataset == 'mnist':
#         return torch.clamp(x, -1.0, 1.0)
#     elif args.dataset == 'gtsrb':
#         return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     else:
#         raise Exception(f'Invalid dataset: {args.dataset}')

def get_clip_image(dataset="cifar10"):
    if dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'cifar10':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'mnist':
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    elif dataset == 'gtsrb':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    else:
        raise Exception(f'Invalid dataset: {args.dataset}')
    return clip_image     

def all2one_target_transform(x, attack_target=1):
    return torch.ones_like(x) * attack_target

def all2all_target_transform(x, num_classes):
    return (x + 1) % num_classes

def get_target_transform(args):
    """Get target transform function
    """
    if args['mode'] == 'all2one':
        target_transform = lambda x: all2one_target_transform(x, args['target_label'])
    elif args['mode'] == 'all2all':
        target_transform = lambda x: all2all_target_transform(x, args['num_classes'])
    else:
        raise Exception(f'Invalid mode {args.mode}')
    return target_transform

def create_trigger_model(dataset, device="cpu", attack_model=None):
    """ Create trigger model """
    if dataset == 'cifar10':
        from attack_models.unet import UNet
        
        atkmodel = UNet(3).to(device)
        # Copy of attack model
        tgtmodel = UNet(3).to(device)
    elif dataset == 'mnist':
        from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
        atkmodel = Autoencoder().to(device)
        # Copy of attack model
        tgtmodel = Autoencoder().to(device)

    elif dataset == 'tiny-imagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb':
        if attack_model is None:
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder().to(device)
            tgtmodel = Autoencoder().to(device)
        elif attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(device)
            tgtmodel = UNet(3).to(device)
    else:
        raise Exception(f'Invalid atk model {dataset}')
    
    return atkmodel, tgtmodel

def create_paths(args):
    if args['mode'] == 'all2one': 
        basepath = os.path.join(args['path'], f"{args['mode']}_{args['target_label']}", args['dataset'], args['clsmodel'])
    else:
        basepath = os.path.join(args['path'], args['mode'], args['dataset'], args['clsmodel'])
   
    basepath = os.path.join(basepath, f"lr{args['lr']}-lratk{args['lr_atk']}-eps{args['eps']}-alpha{args['attack_alpha']}-clsepoch{args['train_epoch']}-atkmodel{args['attack_model']}-atk{args['attack_portion']}")

    if not os.path.exists(basepath):
        print(f'Creating new model training in {basepath}')
        os.makedirs(basepath)
    checkpoint_path = os.path.join(basepath, 'checkpoint.ckpt')
    bestmodel_path = os.path.join(basepath, 'bestmodel.ckpt')
    return basepath, checkpoint_path, bestmodel_path