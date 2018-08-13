import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np


class RetiNet(nn.Module):
    def __init__(self, num_class=2):
        super(RetiNet, self).__init__()
        self.num_class = num_class
        self.in_channel = 3
        # conv1
        self.conv1_1 = nn.Conv2d(self.in_channel, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        # self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn3_3 = nn.BatchNorm2d(256)
        # self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU(inplace=True)
        # self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn4_2 = nn.BatchNorm2d(256)
        # self.relu4_2 = nn.ReLU(inplace=True)
        # self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn4_3 = nn.BatchNorm2d(512)
        # self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        #
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_2 = nn.BatchNorm2d(512)
        # self.relu5_2 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        #
        # self.conv6_1 = nn.Conv2d(256, 256, 3, padding=1)
        # self.bn6_1 = nn.BatchNorm2d(256)
        # self.relu6_1 = nn.ReLU(inplace=True)
        # self.conv6_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn6_2 = nn.BatchNorm2d(512)
        # self.relu6_2 = nn.ReLU(inplace=True)
        # self.pool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/64

        # self.conv7_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn7_1 = nn.BatchNorm2d(512)
        # self.relu7_1 = nn.ReLU(inplace=True)
        # self.conv7_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn7_2 = nn.BatchNorm2d(512)
        # self.relu7_2 = nn.ReLU(inplace=True)
        # self.pool7 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/128

        # self.score_512_to_128 = nn.Conv2d(512, 256, 1, padding=0)
        # self.score_256_to_128 = nn.Conv2d(256, 128, 1, padding=0)

        self.fcn = nn.Sequential(

            nn.Linear(256*16*16, 2048),
            nn.Dropout(),
            nn.Linear(2048, self.num_class),
            # nn.BatchNorm1d(self.num_class),
            # nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.bn1_1(self.conv1_1(h))
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        h = self.relu1_2(h)
        h = self.pool1(h)
        # pool1 = h

        h = self.conv2_1(h)
        h = self.relu2_1(h)
        # h = self.bn2_2(self.conv2_2(h))
        # h = self.relu2_2(h)
        h = self.pool2(h)
        # pool2 = h  # 1/4

        h = self.conv3_1(h)
        h = self.relu3_1(h)
        # h = self.bn3_2(self.conv3_2(h))
        # h = self.relu3_2(h)
        # h = self.bn3_3(self.conv3_3(h))
        # h = self.relu3_3(h)
        h = self.pool3(h)
        # pool3 = h  # 1/8

        h = self.conv4_1(h)
        h = self.relu4_1(h)
        # h = self.bn4_2(self.conv4_2(h))
        # h = self.relu4_2(h)
        # h = self.bn4_3(self.conv4_3(h))
        # h = self.relu4_3(h)
        h = self.pool4(h)

        h = self.conv5_1(h)
        h = self.relu5_1(h)
        # h = self.bn4_2(self.conv4_2(h))
        # h = self.relu4_2(h)
        # h = self.bn4_3(self.conv4_3(h))
        # h = self.relu4_3(h)
        h = self.pool5(h)
        #
        # h = self.bn6_1(self.conv6_1(h))
        # h = self.relu6_1(h)
        # h = self.bn4_2(self.conv4_2(h))
        # h = self.relu4_2(h)
        # h = self.bn4_3(self.conv4_3(h))
        # h = self.relu4_3(h)
        # h = self.pool6(h)

        # h = self.score_256_to_128(h)
        h = h.view(h.size(0), -1)
        h = self.fcn(h)

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# model = RetiNet(num_class=5)
# print(model)
# x = torch.randn(64, 3, 256, 256)
# print(model(Variable(x)).size())
