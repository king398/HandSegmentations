import argparse
import json
from torchvision import transforms
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm

# Custom
from model import HandSegModel
from dataloader import get_dataloader, show_samples, Denorm


def get_dataloaders(args):
    image_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        lambda m: torch.where(m > 0, torch.ones_like(m), torch.zeros_like(m)),
        lambda m: F.one_hot(m[0].to(torch.int64), 2).permute(2, 0, 1).to(torch.float32),
    ])
    dl_args = {
        'data_base_path': args.data_base_path,
        'datasets': args.datasets.split(' '),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'batch_size': args.batch_size,
    }
    dl_train = get_dataloader(**dl_args, partition='train', shuffle=True)
    dl_validation = get_dataloader(**dl_args, partition='validation', shuffle=False)
    dl_test = get_dataloader(**dl_args, partition='test', shuffle=False)
    dls = {
        'train': dl_train,
        'validation': dl_validation,
        'test': dl_test,
    }
    return dls
