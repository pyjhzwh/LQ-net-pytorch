#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from lqnet import lq_act

'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsample(data, outsize=28):
    data = F.interpolate(data,outsize, mode = 'bilinear')
    #print(data.size(),"=?",outsize)
    return data
'''

class convbnrelu_block(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=0,
            stride = 1, relu = True, usebn= True, early_predict=0, bn_adjust=0,
            quantAct=False, bits=8, key=''):
        super(convbnrelu_block, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # early_predict mode
        # 0: no early_predict
        # 1: exact mode (first positive weights, then negative weights from the MSB to LSB)
        # 2: predictive mode ( deal with postive and negative weights together)
        self.early_predict = early_predict
        self.bn_adjust = bn_adjust
        self.usebn = usebn
        if self.usebn:
            useconvbias = False
        else:
            useconvbias = True
        self.quantAct = quantAct
        self.bits=bits

        if self.quantAct:
            self.lqAct = lq_act(key, self.bits)
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, self.kernel_size,
                padding=self.padding, stride=self.stride, bias = useconvbias)
        
        if self.usebn is True:
            self.bn = nn.BatchNorm2d(self.out_planes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, stats=None):

        #if self.early_predict == 0:
            #if epoch <= 250:
            #    return(self.relu(self.bn(self.conv(x))+self.bn_adjust)), computation
            #else:
        if self.quantAct and stats is not None:
            x = self.lqAct.update(x, stats, test=not self.training)
        if self.usebn is True:
            return (self.relu(self.bn(self.conv(x))))
        else:
            return self.relu(self.conv(x))
    

class convrelubn_block(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=0,
            stride = 1, relu = True,):
        super(convrelubn_block, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride


        self.conv = nn.Conv2d(self.in_planes, self.out_planes, self.kernel_size,
                padding=self.padding, stride=self.stride, bias = False)

        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(self.out_planes)
    
    def forward(self, x,epoch=0):

        return(self.bn(self.relu(self.conv(x))))

class convbnresrelu_block(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=0,
            stride = 1, relu = True, early_predict=0, bn_adjust=0):
        super(convbnresrelu_block, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # early_predict mode
        # 0: no early_predict
        # 1: exact mode (first positive weights, then negative weights from the MSB to LSB)
        # 2: predictive mode ( deal with postive and negative weights together)
        self.early_predict = early_predict
        self.bn_adjust = bn_adjust

        self.conv = nn.Conv2d(self.in_planes, self.out_planes, self.kernel_size,
                padding=self.padding, stride=self.stride, bias = False)
        
        self.bn = nn.BatchNorm2d(self.out_planes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, identity, epoch=0):

        #if self.early_predict == 0:
            #if epoch <= 250:
            #    return(self.relu(self.bn(self.conv(x))+self.bn_adjust)), computation
            #else:
        return (self.relu(identity + self.bn(self.conv(x))))

class convbnsilu_block(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=0,
            stride = 1, swish = True, usebn= True, early_predict=0, bn_adjust=0,
            quantAct=False, bits=8, key=''):
        super(convbnsilu_block, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # early_predict mode
        # 0: no early_predict
        # 1: exact mode (first positive weights, then negative weights from the MSB to LSB)
        # 2: predictive mode ( deal with postive and negative weights together)
        self.early_predict = early_predict
        self.bn_adjust = bn_adjust
        self.usebn = usebn
        if self.usebn:
            useconvbias = False
        else:
            useconvbias = True
        self.quantAct = quantAct
        self.bits=bits

        if self.quantAct:
            self.lqAct = lq_act(key, self.bits)
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, self.kernel_size,
                padding=self.padding, stride=self.stride, bias = useconvbias)
        
        if self.usebn is True:
            self.bn = nn.BatchNorm2d(self.out_planes)

        self.silu = nn.SiLU(inplace=True)
    
    def forward(self, x, stats=None):

        #if self.early_predict == 0:
            #if epoch <= 250:
            #    return(self.relu(self.bn(self.conv(x))+self.bn_adjust)), computation
            #else:
        if self.quantAct and stats is not None:
            x = self.lqAct.update(x, stats, test=not self.training)
        if self.usebn is True:
            return (self.silu(self.bn(self.conv(x))))
        else:
            return self.silu(self.conv(x))