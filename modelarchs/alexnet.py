#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ._conv_block import *
from ._fc_block import *
import itertools
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
      
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Alexnet(nn.Module):
    def __init__(self, nclass=1000,block_type='convbnrelu', pretrained=False):
        super(Alexnet,self).__init__()
        self.nclass = nclass
        self.scale = np.zeros(4, dtype=float)
        self.zero_point = np.zeros(4, dtype=int)
        # if use Qint to compute inference rather than scale * (Q - zero_point) -- float

        if block_type == 'convbnrelu':
            conv_block = convbnrelu_block 
        elif block_type == 'convrelubn':
            conv_block = convrelubn_block
        else:
            ValueError('conv_block type either be convbnreku or convrelubn')

        self.conv0 = conv_block(3, 64, kernel_size=11, padding=2, stride=4, relu=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.conv0 = conv_block(3, 96, kernel_size=3, padding=1, stride=1, relu=True,early_predict=self.early_predict, b_thr=b_thr[0], bits=bits, addbias=addbias, adjustthr=adjustthr,kernel_wise=False,quantW = self.quantW,quantAct = self.quantAct)
        self.conv1 = conv_block(64, 192, kernel_size=5, padding=2, stride=1, relu=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = conv_block(192, 384, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv3 = conv_block(384, 256, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv4 = conv_block(256, 256, kernel_size=3, padding=1, stride=1, relu=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.dropout0 = nn.Dropout()
        self.fc5 = fc_block(256 * 6 * 6, 4096)
        self.dropout1 = nn.Dropout()
        self.fc6 = fc_block(4096, 4096)
        self.fc7 = nn.Linear(4096, nclass)
        '''
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nclass),
        )
        '''

        if pretrained is False:
            print('init conv and bn')
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
        else:
            print('init bn')
            for name, m  in self.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()



    def forward(self,x):

        computation = torch.zeros(4, dtype=torch.float).cuda()
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.conv1(x) 
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout0(x)
        x = self.fc5(x)
        x = self.dropout1(x)
        x = self.fc6(x)
        x = self.fc7(x)
        #x = self.classifier(x)

        return x
    


def alexnet(pretrained: bool = False, progress: bool = True,**kwargs) -> Alexnet:

    model = Alexnet(pretrained = pretrained, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                            progress=progress)
        new_key_dict={"0":"conv0", "3":"conv1",  "6":"conv2", "8": "conv3", "10": "conv4"}
        fc_new_key_dict = {"1": "fc5", "4": "fc6", "6": "fc7"}
        for key in list(state_dict.keys()):
            if 'features' in key and 'weight' in key:
                #print(state_dict[key][0,0,0,:5])
                state_dict[new_key_dict[key.split(".")[1]]+".conv.weight"] = state_dict.pop(key)
            if 'classifier' in key:
                if key.split(".")[1] is not "6":
                    state_dict[fc_new_key_dict[key.split(".")[1]]+".fc."+key.split(".")[2]] = state_dict.pop(key)
                else:
                    state_dict[fc_new_key_dict[key.split(".")[1]]+"."+key.split(".")[2]] = state_dict.pop(key)
        model.load_state_dict(state_dict,strict=False)
        #for key, params in model.named_parameters():
        #    if ".conv.weight" in key:
        #        print(key)
        #        print(params[0,0,0,:5])
    return model
