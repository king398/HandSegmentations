import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import pandas as pd


def make_transparent_foreground(pic, mask):
    # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # add an alpha channel with and fill all with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
    # merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
    # create a transparent background
    bg = np.zeros(alpha_im.shape)
    # setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground


def remove_background(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image, bin_mask)

    return foreground, bin_mask


def inference(img):
    foreground, _ = remove_background(img)
    return foreground


model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', weights='DeepLabV3_ResNet101_Weights.DEFAULT')
model.eval()

paths = glob.glob('/home/mithil/PycharmProjects/HandWriting/data/asl_alphabet_train/*/*.jpg')
ids = []
labels = []
for i, x in tqdm(enumerate(paths)):
    rgb = cv2.imread(x)

    cv2.imwrite(f"/home/mithil/PycharmProjects/HandWriting/data/train_sign/{i}.jpeg", rgb)
    label = x.split('/')[-2]
    ids.append(i)
    labels.append(label)
df = pd.DataFrame.from_dict(
    {
        'ids': ids,
        'labels': labels
    }
)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('/home/mithil/PycharmProjects/HandWriting/data/train_sign.csv', index=False)
