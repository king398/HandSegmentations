import cv2
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from matplotlib.patches import Rectangle

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_model():
    model = smp.Unet(
        encoder_name='efficientnet-b0',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to('cuda')
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype('float32')  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')
    plt.show()


def transform():
    return A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        ToTensorV2()
    ], p=1.0)


transform = transform()
model = load_model(f"/home/mithil/PycharmProjects/HandWriting/models/best_epoch.bin")


def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def plot_batch(image_path):
    img = load_img(image_path)
    image = transform(image=img)['image']
    image = image.unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(image)
        pred = (nn.Sigmoid()(pred) > 0.5).double().cpu().detach()
    for idx in range(1):
        msk = pred[idx,].permute((1, 2, 0)).numpy() * 255.0
        show_img(msk)

    plt.tight_layout()
    plt.show()
    mask = msk.squeeze(2)
    mask[mask != 0] = 255

    image = cv2.resize(img, (224, 224))
    ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    image[thresh1 == 0] = 0
    image = crop(image)
    image = cv2.resize(image, (224, 224))
    return image


cropped_image = plot_batch('/home/mithil/PycharmProjects/HandWriting/hand_removed_backgroud.jpeg')

plt.imshow(cropped_image)
plt.show()
