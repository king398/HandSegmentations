import cv2
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np
from torch import nn


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


BCELoss = smp.losses.SoftBCEWithLogitsLoss()

model = load_model('/home/mithil/PycharmProjects/HandWriting/models/best_epoch.bin')
image = cv2.imread('/home/mithil/PycharmProjects/HandWriting/data/archive/train/train/a/images/000000.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread('/home/mithil/PycharmProjects/HandWriting/test_mask.png',
                  cv2.IMREAD_GRAYSCALE)
mask = mask[np.newaxis, np.newaxis, :, :]
image = np.transpose(image, (2, 0, 1))
image = torch.tensor(image, dtype=torch.float).cuda().unsqueeze(0)
mask = torch.tensor(mask, dtype=torch.float).cuda()
y_pred = model(image)
pred = (nn.Sigmoid()(y_pred) > 0.5).double()
pred = y_pred.detach().cpu().numpy()
pred = pred.squeeze(0)
pred = pred.transpose()
pred = pred.astype(np.uint8) * 255
pred[pred != 0] = 255
cv2.imshow("pred", pred)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
mask = cv2.imread('/home/mithil/PycharmProjects/HandWriting/data/train/mask/3.png',
                  cv2.IMREAD_GRAYSCALE)
mask[mask != 0] = 255

# (thresh, blackAndWhiteImage) = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("mask", mask)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

image = cv2.imread('/home/mithil/PycharmProjects/HandWriting/data/train/image/3.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


image_cropped = get_segment_crop(image, mask=mask)
cv2.imshow("image_cropped", image_cropped)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
