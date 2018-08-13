# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import DataParallel

import numpy as np
from PIL import Image
import cv2
import pandas as pd
from pandas import Series
import csv
from models import densenet121

valFolder = '/dev/shm/dr_datasets_test'
csvPath = 'test.csv'
csvoutPath = 'test_out_epoch_121.csv'
resume = 'epoch_121_densenet121_finetune_1024_1024_classes_2.ckpt'
width = 1024
height= 1024

def ReadLabelFile(labelPath):
    csv = pd.read_csv(labelPath)
    level = Series.as_matrix(csv['level'])
    files = Series.as_matrix(csv['image'])
    return dict(zip(files, level))

def GetPreprocess():
    return transforms.Compose([
        transforms.Resize((width, height), interpolation=2),
        transforms.ToTensor(),
    ])


if __name__ == "__main__":
    import sys

    model = densenet121(pretrained=False)
    # fc_features = model.fc.in_features 138448
    model.classifier = nn.Linear(model.classifier.in_features, 2, bias=True)
    if torch.cuda.is_available():
        model.cuda()

    checkpointpath = '../checkpoint/'
    if os.path.isfile(os.path.join(checkpointpath, resume)):
        print("loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(os.path.join(checkpointpath, resume))
        val_acc = checkpoint['acc']
        print('model accuracy is ',  val_acc)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("no checkpoint found at '{}'".format(os.path.join(checkpointpath, resume)))

    if not model:
        print('model read failed.')
        sys.exit()

    model = model.cuda()
    model.eval()

    if not os.path.isdir(valFolder):
        print('validation folder not exist.')
        sys.exit()

    with open(csvoutPath, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['file', 'groundtruth', 'predict', 'prob0', 'prob1'])

        df = pd.read_csv(csvPath)
        for i in range(len(df)):
            filename = str(df['image'][i]) + '.jpg'
            groundTruth = df['level'][i]
            if groundTruth > 0:
                groundTruth = 1
            filenpath = os.path.join(valFolder, filename)
            if not os.path.exists(filenpath):
                continue
            cvImg = cv2.imread(filenpath)
            img_pil = Image.fromarray(cvImg)

            preprocess = GetPreprocess()
            var = Variable(preprocess(img_pil).unsqueeze(0)).cuda()
            logit = model(var)

            h_x = F.softmax(logit, dim=1).data.squeeze()
            # probs, idx = h_x.sort(0, True)

            prob, predict = torch.max(logit.data, 1)

            f_csv.writerow([filename, int(groundTruth), predict.cpu().numpy()[0], h_x.cpu().numpy()[0], h_x.cpu().numpy()[1]])
            print(i)
