#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from ._conv_block import *


class all_cnn_net(nn.Module):
    def __init__(self, nclass=10, block_type='convbnrelu'):
        super(all_cnn_net,self).__init__()
        self.nclass = nclass

        if block_type == 'convbnrelu':
            conv_block = convbnrelu_block 
        elif block_type == 'convrelubn':
            conv_block = convrelubn_block
        else:
            ValueError('conv_block type either be convbnreku or convrelubn')

        self.conv0 = conv_block(3, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv1 = conv_block(96, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv2 = conv_block(96, 96, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout0 = nn.Dropout(p=0.5)

        self.conv3 = conv_block(96, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv4 = conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv5 = conv_block(192, 192, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout1 = nn.Dropout(p=0.5)

        self.conv6 = conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv7 = conv_block(192, 192, kernel_size=1, padding=0, stride=1, relu=True)
        self.conv8 = conv_block(192, 10, kernel_size=1, padding=0, stride=1, relu=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.normal_(m.weight)
                #nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #nn.init.constant_(m.weight,1)
                #nn.init.constant_(m.bias,0)


    def forward(self,x,epoch):

        x = self.conv0(x,epoch)
        x = self.conv1(x,epoch)
        x = self.conv2(x,epoch)

        x = self.dropout0(x)

        x = self.conv3(x,epoch)
        x = self.conv4(x,epoch)
        x = self.conv5(x,epoch)

        x = self.dropout1(x)

        x = self.conv6(x,epoch)
        x = self.conv7(x,epoch)
        x = self.conv8(x,epoch)

        #x = self.avgpool(x)
        x = F.avg_pool2d(x, kernel_size=x.size(2))
        x = x.view(x.size(0),-1)
        
        return x
    