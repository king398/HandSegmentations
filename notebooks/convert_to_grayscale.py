import glob
import cv2
from tqdm import tqdm
import pandas as pd

mask_path = glob.glob('/home/mithil/PycharmProjects/HandWriting/data/archive/train/train/*/segmentation/*.png')
labels = []
ids = []
for x, i in tqdm(enumerate(mask_path)):
    mask = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    mask[mask != 0] = 255

    i = i.split('/')

    cv2.imwrite(f'/home/mithil/PycharmProjects/HandWriting/data/train/mask/{x}.png', mask)
    label = i[-3]
    id = x
    labels.append(label)
    ids.append(id)
    image = cv2.imread(f'/home/mithil/PycharmProjects/HandWriting/data/archive/train/train/{label}/images/{i[-1]}')
    cv2.imwrite(f'/home/mithil/PycharmProjects/HandWriting/data/train/image/{x}.png', image)

train_df = pd.DataFrame.from_dict({
    'ids': range(20820),

})
print(train_df.head())
train_df.to_csv('/home/mithil/PycharmProjects/HandWriting/data/train/train.csv', index=False)
