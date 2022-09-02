from tkinter import Tk, Label, StringVar
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import timm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import preprocessing

from torch.cuda.amp import autocast
import time


def get_valid_transforms(dim):
    return Compose(
        [

            Normalize(),
            ToTensorV2(),
        ]
    )


class Model(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, in_chans=3, num_classes=29)

    def forward(self, x):
        output = self.model(x)

        return output


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
        with autocast():
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

background = None

classification_model = Model('tf_efficientnet_b0_ns')
classification_model.load_state_dict(
    torch.load('/home/mithil/PycharmProjects/HandWriting/models/resnet50_epoch_9.pth'))
classification_model.eval()
classification_model.cuda()
transforms_classification = get_valid_transforms(384)
label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.load('/home/mithil/PycharmProjects/HandWriting/models/classes.npy', allow_pickle=True)
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))


# import the opencv library
def classify(image):
    image = transforms_classification(image=image)['image']
    image = image.to('cuda', non_blocking=True).float()
    image = image.unsqueeze(0)
    with torch.no_grad():
        with autocast():
            probability = classification_model(image).float().softmax(1)
            output = probability.argmax(dim=1).detach().cpu().numpy()
            probability = probability.detach().cpu().numpy().squeeze(0)
    return label_encoder.inverse_transform(output), probability[output]


# define a video capture object
vid = cv2.VideoCapture(0)
window = Tk()
window.title('Segment')
var = StringVar()

label = Label(window, height=69, width=69, padx=2, pady=2,
              textvariable=var, font=("Arial", 69))
label.pack()
target_layer = classification_model
i = 0

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    frame = cv2.resize(frame, (512, 512))
    fore = inference(Image.fromarray(frame))
    image = cv2.cvtColor(fore, cv2.COLOR_BGRA2RGB)

    label, probability = classify(image)

    # Display the resulting frame
    frame = np.vstack((frame, cv2.cvtColor(fore, cv2.COLOR_BGRA2BGR)))
    cv2.imshow("frame", frame)
    var.set(f"Label {label[0]}")

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
    window.update_idletasks()
    window.update()

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
