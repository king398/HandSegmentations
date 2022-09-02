import gc
import os
import random
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import pandas as pd
# Deep learning Stuff
import yaml
from sklearn import preprocessing
from torch.optim import *
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def return_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def return_filpath(name, folder):
    path = os.path.join(folder, f'{name}.jpeg')
    return path


class Model(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=29)

    def forward(self, x):
        output = self.model(x)

        return output


def accuracy_score(output, labels):
    output = output.detach().cpu()
    labels = labels.detach().cpu()
    output = torch.softmax(output, 1)
    accuracy = (output.argmax(dim=1) == labels).float().mean()
    accuracy = accuracy.detach().cpu().numpy()
    return accuracy


def get_scheduler(optimizer):
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        eta_min=1e-3,
        last_epoch=-1
    )


def train_fn(train_loader, model, criterion, optimizer, epoch, scaler, scheduler=None):
    """Train a model on the given image using the given parameters .
    Args:
        train_loader ([DataLoader]): A pytorch dataloader that contains train.yaml images and returns images,target
        model ([Module]): A pytorch model
        criterion ([Module]): Pytorch loss
        optimizer ([Optimizer]): [description]
        epoch ([int]): [description]
        cfg ([dict]): [description]
        scheduler ([Scheduler], optional): [description]. Defaults to None.
        fold ([int]): Fold Training
    """
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    outputs = None
    targets = None

    for i, (images, target) in enumerate(stream, start=1):
        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        target = target.to(device)

        with autocast():
            output = model(images).float()

            loss = criterion(output, target)
            accuracy = accuracy_score(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("accuracy", accuracy)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")
        if outputs is None and targets is None:
            outputs = output
            targets = target
        else:
            outputs = torch.cat([outputs, output], dim=0)
            targets = torch.cat([targets, target], dim=0)


def validate_fn(val_loader, model, criterion, epoch):
    device = torch.device('cuda')
    metric_monitor = MetricMonitor()

    model.eval()
    stream = tqdm(val_loader)
    outputs = None
    targets = None
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):

            images = images.to(device, non_blocking=True)
            target = target.to(device)

            with autocast():
                output = model(images).float()

            loss = criterion(output, target)
            accuracy = accuracy_score(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("accuracy", accuracy)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
            if outputs is None and targets is None:
                outputs = output
                targets = target
            else:
                outputs = torch.cat([outputs, output], dim=0)
                targets = torch.cat([targets, target], dim=0)


class Mayo_Dataset(Dataset):

    def __init__(self, image_path, targets, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.targets[idx]).long()
        return image, label


def get_train_transforms(dim):
    return Compose(
        [
            RandomResizedCrop(height=dim, width=dim),
            HorizontalFlip(),
            VerticalFlip(),

            Normalize(),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(dim):
    return Compose(
        [
            Resize(height=dim, width=dim),

            Normalize(),
            ToTensorV2(),
        ]
    )


label_encoder = preprocessing.LabelEncoder()

df = pd.read_csv('/home/mithil/PycharmProjects/HandWriting/data/train_sign.csv')
df['label'] = label_encoder.fit_transform(df['labels'])
df['file_path'] = df['ids'].apply(
    lambda x: return_filpath(x, folder='/home/mithil/PycharmProjects/HandWriting/data/train_sign'))
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
gc.enable()
device = return_device()
seed_everything(42)
model = Model('tf_efficientnet_b2_ns')

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

scaler = torch.cuda.amp.GradScaler()
train_dataset = Mayo_Dataset(
    image_path=train_df['file_path'],
    targets=train_df['label'],
    transform=get_train_transforms(384)

)
valid_dataset = Mayo_Dataset(
    image_path=valid_df['file_path'],
    targets=valid_df['label'],
    transform=get_valid_transforms(384)

)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=15, pin_memory=True
)

val_loader = DataLoader(
    valid_dataset, batch_size=16 * 2, shuffle=False,
    num_workers=15, pin_memory=True
)
scheduler = get_scheduler(optimizer)
model_name = None
for epoch in range(10):
    train_fn(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
    validate_fn(val_loader, model, criterion, epoch)
    if model_name is not None:
        os.remove(model_name)
    torch.save(model.state_dict(),
               f"/home/mithil/PycharmProjects/HandWriting/models/effnet_b2_epoch_{epoch}.pth")
    model_name = f"/home/mithil/PycharmProjects/HandWriting/models/effnet_b2_epoch_{epoch}.pth"
gc.collect()
torch.cuda.empty_cache()
del train_dataset
del valid_dataset
del train_loader
del val_loader
del model
del optimizer
del scheduler
torch.cuda.empty_cache()
