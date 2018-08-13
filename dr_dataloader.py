import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from PIL import ImageFilter
import numpy as np


class DRDataLoader(data.Dataset):
    def __init__(self, csv_path, img_path, transform=None, train=True):
        df = pd.read_csv(csv_path)
        if train:
            # result = df.sample(n=20000, random_state=2009)
            df_0 = df[df['level'] == 0]
            df_1 = df[df['level'] > 0]
            result = pd.concat([df_0.sample(n=7500, random_state=2008), df_1], axis=0).sample(frac=1)

            # result = df
        else:
            # result = df.sample(n=500, random_state=2005)
            result = df

        self.transform = transform
        self.img_path = img_path
        self.X_train = np.array(result['image'])
        self.y_train = np.array(result['level'], dtype=np.int64)

    def __getitem__(self, index):
        path = self.img_path + str(self.X_train[index]) + '.jpg'

        if os.path.isfile(path):
            img = Image.open(path)
            # if index > 9000:
            #     img.filter(ImageFilter.SHARPEN)
            label = self.y_train[index]
            image_name = str(self.X_train[index]) + '.jpg'
        else:
            img = Image.open(self.img_path + str(self.X_train[1]) + '.jpg')
            label = self.y_train[1]
            image_name = str(self.X_train[1]) + '.jpg'

        # 二分类
        if label > 0:
            label = np.int64(1)
        else:
            label = np.int64(0)

        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, image_name

    def __len__(self):
        return self.X_train.size
