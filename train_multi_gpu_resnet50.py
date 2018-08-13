# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms, models
import time
import os
from dr_dataloader import DRDataLoader
from torch.optim import lr_scheduler

from models.resnet import resnet50

import logging  # 引入logging模块
from torch.nn import DataParallel
# from cnn_finetune import make_model


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

width = 512
height= 512

def train_and_val(args):
    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
    # 第一步，创建一个logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Diabetic_Reinopathy_Detection/Logs/'
    log_name = log_path + rq + '.log'
    logfile = log_name
    handler = logging.FileHandler(logfile, 'a')
    handler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(handler)

    # writer = SummaryWriter(comment='')

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),  # 从原图像随机切割一张（224， 224）的图像
        # transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转
        transforms.RandomVerticalFlip(),    # 以0.5的概率垂直翻转
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  #HSV以及对比度变化
        transforms.RandomAffine(45),
        transforms.RandomGrayscale(),
        transforms.RandomRotation(10),      # 在（-10， 10）范围内旋转
        transforms.Resize((width, height), interpolation=2),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((width, height), interpolation=2),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    train_ds = DRDataLoader('csv/train_balance_50000.csv', '/root/lg/dr_datasets_1024/', transform=train_transform, train=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=RandomSampler(train_ds))

    val_ds = DRDataLoader('csv/test_balance_700.csv', '/root/lg/dr_datasets_1024/', transform=val_transform, train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.test_batch_size, sampler=RandomSampler(val_ds))

    model = resnet50()

    # print(model)
    # model = DataParallel(model)
    criterion = nn.CrossEntropyLoss(size_average=True)
    #
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if not args.disable_cuda and torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    # Observe that all parameters are being optimized

    if args.resume:
        if os.path.isfile(os.path.join('./checkpoint', args.resume)):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join('./checkpoint', args.resume))
            args.start_epoch = checkpoint['epoch']
            args.acc = checkpoint['acc']
            print('epoch', args.start_epoch, 'acc =', args.acc)

            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            # for index, param in enumerate(model.parameters()):
            #     param.requires_grad = False  # 冻结参数的更新
            #
            # for index, param in enumerate(model.decoder.parameters()):
            #     param.requires_grad = True  # 冻结参数的更新

            # model.load_state_dict(checkpoint['state_dict'])

            logger.debug("loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))
            logger.debug("loaded checkpoint '{}')".format(args.resume))
    else:
        logger.debug('checkping is none \n')
        logger.debug("loaded checkpoint '{}')".format(args.resume))


    for epoch in range(args.start_epoch + 1, args.epochs):

        model.train()
        exp_lr_scheduler.step()
        train_loss = []
        train_correct = 0
        train_total = 0
            # Iterate over data.
        for idx, (inputs, target) in enumerate(train_loader):
            if not args.disable_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            inputs = Variable(inputs)
            target = Variable(target)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # import numpy as np
            # aa = np.log(np.sum(np.exp(outputs.data.cpu().numpy()), axis=1))
            # myloss = -outputs.data.cpu().numpy()[0, target.data.cpu().numpy()] + aa

            _, preds = torch.max(outputs.data, 1)

            # statistics
            train_loss.append(loss.data[0])
            train_total += inputs.size()[0]
            train_correct += (preds == target.data).sum()

            # ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

            if idx % (args.interval-1) == 0:
                print('total', train_total, '    trian_correct', train_correct)

                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.5f}  Acc: {:.3f}'.format(
                    epoch, idx*len(inputs), len(train_loader.dataset), 100.*idx / len(train_loader),
                    loss.data[0],
                    train_correct * 1.0 / train_total)
                )
                s = 'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.5f}  Acc: {:.3f}'.format(
                    epoch, idx*len(inputs), len(train_loader.dataset), 100.*idx / len(train_loader),
                    loss.data[0],
                    train_correct * 1.0 / train_total)
                logger.debug(s)

        train_epoch_loss = sum(train_loss)/ len(train_loss)
        train_epoch_acc = 1.0 * train_correct / train_total

        # writer.add_scalar('train' + '/epoch_loss', epoch_loss, epoch)
        # writer.add_scalar('train' + '/epoch_acc', epoch_acc, epoch)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Train', train_epoch_loss, train_epoch_acc))
        logger.debug('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Train', train_epoch_loss, train_epoch_acc))


        model.eval()
        val_loss = []
        val_correct = 0
        val_total = 0

        for idx, (inputs, target) in enumerate(val_loader):
            if not args.disable_cuda and torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            inputs = Variable(inputs)
            target = Variable(target)
            outputs = model(inputs)

            loss = criterion(outputs, target)

            _, preds = torch.max(outputs.data, 1)
            # statistics
            val_loss.append(loss.data[0])
            val_total += inputs.size()[0]
            val_correct += (preds == target.data).sum()
    #
        val_epoch_loss = sum(val_loss) / len(val_loss)
        val_epoch_acc = 1.0 * val_correct / val_total

        # logger.debug('*********************************************')
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Validate', val_epoch_loss, val_epoch_acc))

        logger.debug('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Validate', val_epoch_loss, val_epoch_acc))

        # if val_epoch_acc > args.acc and train_epoch_acc > 0.5:
        if epoch % 10 == 0:
            logger.debug('Saving...' + str(epoch))

            state = {
                # 'net': net.module if not args.cuda else net,
                'acc': val_epoch_acc,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if not os.path.isdir('./checkpoint'):
                os.mkdir('./checkpoint')
            resume = args.resume + str(epoch)
            torch.save(state, os.path.join('./checkpoint', resume))
            args.acc = val_epoch_acc
            print('Saving...' + str(epoch))

def test():
    import csv
    import cv2
    from PIL import Image
    import sys
    import pandas as pd

    labelPath = '../checkpoint/validate_1250.csv'
    outputPath = 'inception_v3_finetune_9000_512_512_output.csv'
    validationFolder = '/root/lg/DR_datasets_1024/'

    val_transform = transforms.Compose([
        transforms.Resize((width, height), interpolation=2),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    # val_ds = DRDataLoader('validate_1250.csv', '/root/lg/DR_datasets_1024/', transform=val_transform, train=False)
    # val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.test_batch_size, sampler=RandomSampler(val_ds))

    # model = make_model('inception_v3', num_classes=5, pretrained=False, input_size=(width, height))
    model = make_model('resnet18', num_classes=5, pretrained=True, input_size=(width, height), dropout_p=0.5)

    criterion = nn.CrossEntropyLoss(size_average=True)

    if not args.disable_cuda and torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    if args.resume:
        if os.path.isfile(os.path.join('./checkpoint', args.resume)):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join('./checkpoint', args.resume))
            args.start_epoch = checkpoint['epoch']
            args.acc = checkpoint['acc']
            print('acc=', args.acc)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("no checkpoint found at '{}'".format(args.resume))
    else:
        print('checkping is none \n')

    model.eval()

    if not os.path.isdir(validationFolder):
        print('validation folder not exist.')
        sys.exit()

    with open(outputPath, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['file', 'groundtruth', 'predict', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4'])

        df = pd.read_csv(labelPath)
        for i in range(len(df)):
            filename = str(df['image'][i]) + '.jpg'
            groundTruth = df['level'][i]

            filenpath = os.path.join(validationFolder, filename)
            if not os.path.exists(filenpath):
                continue
            cvImg = cv2.imread(filenpath)
            img_pil = Image.fromarray(cvImg)

            # preprocess = GetPreprocess()
            var = Variable(val_transform(img_pil).unsqueeze(0).cuda())
            logit = model(var)

            h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()
            # probs, idx = h_x.sort(0, True)

            prob, predict = torch.max(logit.data, 1)

            f_csv.writerow([filename, groundTruth, predict.item(), h_x[0].item(), h_x[1].item(),h_x[2].item(),h_x[3].item(),h_x[4].item()])
            print(i)

def write_to_csv(epoch_name, val_acc):
    pass

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--acc', type=float, default=0, metavar='M',
                        help='acc (default: 0.)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default='myresnet_1024_1024_classes_2.ckpt', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    train_and_val(args)

#
# # test中合成图像使用
# import PIL.Image as Image
# import numpy as np
#
# a = np.load('A.npy')
# a0 = a[0]
# r = Image.fromarray(np.uint8(a[0]*255)).convert('L')
# g = Image.fromarray(np.uint8(a[1]*255)).convert('L')
# b = Image.fromarray(np.uint8(a[2]*255)).convert('L')
# image = Image.merge("RGB", (r, g, b))
# # 显示图片
# plt.imshow(image)
# plt.show()

