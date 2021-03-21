#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from ._conv_block import *


class all_cnn_net(nn.Module):
    def __init__(self, nclass=10, block_type='convbnrelu', quantAct=False, bits=[None]*7):
        super(all_cnn_net,self).__init__()
        self.nclass = nclass
        self.bits=bits

        if block_type == 'convbnrelu':
            conv_block = convbnrelu_block 
        elif block_type == 'convrelubn':
            conv_block = convrelubn_block
        else:
            ValueError('conv_block type either be convbnreku or convrelubn')

        self.conv0 = conv_block(3, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv1 = conv_block(96, 96, kernel_size=3, padding=1, stride=1, relu=True, 
                                quantAct=quantAct, bits=self.bits[0], key='conv1')
        self.conv2 = conv_block(96, 96, kernel_size=3, padding=1, stride=2, relu=True, 
                                quantAct=quantAct, bits=self.bits[1], key='conv2')

        self.dropout0 = nn.Dropout(p=0.5)

        self.conv3 = conv_block(96, 192, kernel_size=3, padding=1, stride=1, relu=True, 
                                quantAct=quantAct, bits=self.bits[2], key='conv3')
        self.conv4 = conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True, 
                                quantAct=quantAct, bits=self.bits[3], key='conv4')
        self.conv5 = conv_block(192, 192, kernel_size=3, padding=1, stride=2, relu=True, 
                                quantAct=quantAct, bits=self.bits[4], key='conv5')

        self.dropout1 = nn.Dropout(p=0.5)

        self.conv6 = conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True, 
                                quantAct=quantAct, bits=self.bits[5], key='conv6')
        self.conv7 = conv_block(192, 192, kernel_size=1, padding=0, stride=1, relu=True, 
                                quantAct=quantAct, bits=self.bits[6], key='conv7')
        self.conv8 = conv_block(192, 10, kernel_size=1, padding=0, stride=1, relu=True)


        self.target_acts = [self.conv1, self.conv2, self.conv3, self.conv4, 
                            self.conv5, self.conv6, self.conv7]

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


    def forward(self,x,stats=None):

        x = self.conv0(x)
        x = self.conv1(x,stats)
        x = self.conv2(x,stats)

        x = self.dropout0(x)

        x = self.conv3(x,stats)
        x = self.conv4(x,stats)
        x = self.conv5(x,stats)

        x = self.dropout1(x)

        x = self.conv6(x,stats)
        x = self.conv7(x,stats)
        x = self.conv8(x)

        #x = self.avgpool(x)
        x = F.avg_pool2d(x, kernel_size=x.size(2))
        x = x.view(x.size(0),-1)
        
        return x

    def print_Actinfo(self):

        for layer in self.target_acts:
            layer.lqAct.print_info()

    