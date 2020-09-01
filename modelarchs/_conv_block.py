#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

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

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=1,
            stride = 1, relu = True, early_predict=0, bn_adjust=0):
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

        self.conv = nn.Conv2d(self.in_planes, self.out_planes, self.kernel_size,
                padding=self.padding, stride=self.stride, bias = False)
        
        self.bn = nn.BatchNorm2d(self.out_planes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x,epoch=0):
        computation = 0

        if self.early_predict == 0:
            if epoch <= 250:
                return(self.relu(self.bn(self.conv(x))+self.bn_adjust)), computation
            else:
                return(self.relu(self.bn(self.conv(x)))), computation

        elif self.early_predict == 1:
            # groudtruth conv
            ground_conv = self.conv(x)
            ground_relu = self.relu(ground_conv)
            # early_predict conv
            saved_w = self.conv.weight.data
            basis = torch.min(saved_w.abs())
            bits = torch.log2((torch.max(saved_w) / basis)+1).int()
            #print('basis',basis,' bits',bits)
            mask_pos = (saved_w >= 0).float()
            self.conv.weight.data = saved_w * mask_pos
            conv = self.conv(x)
            # computation for all postive weights
            computation = torch.sum(mask_pos) / saved_w.numel()
            remainder = saved_w.detach() * (1-mask_pos) / basis
            saved_w_test = self.conv.weight.data
            #print('remainder',remainder[0,0,:,:])
            for i in range(bits):
                # mask for negative weights, bit by bit
                mask_wn = torch.round(remainder / 2**(bits-1-i))
                remainder = remainder - mask_wn * (2**(bits-1-i))
                # mask for output becomes featuremap negative
                mask_on = (conv > 0).float()
                self.conv.weight.data = basis * mask_wn * (2**(bits-1-i))
                saved_w_test += self.conv.weight.data
                conv += self.conv(x) * mask_on
                # computation do not count if output featuremap has already been negative
                # consider sparcity of weights
                #computation += (-1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on) / conv.numel() / bits.float()
                # exclude sparcity of weights
                computation += torch.sum(1-mask_pos) / saved_w.numel() * torch.sum(mask_on) / conv.numel() / bits.float()
            #print(torch.sum(conv>0).float()/ conv.numel())
            x = self.relu(conv)
            self.conv.weight.data = saved_w

            if torch.max(x-ground_relu) > 1e-3:
                print("different result of relu")

            return x, computation

        elif self.early_predict == 2:

            # groudtruth conv
            ground_conv = self.conv(x)
            ground_relu = self.relu(ground_conv)
            # early_predict conv
            saved_w = self.conv.weight.data
            basis = torch.min(saved_w.abs())
            bits = torch.log2((torch.max(saved_w) / basis)+1).int()
            #print('basis',basis,' bits',bits)
            mask_pos = (saved_w >= 0).float()
            conv = torch.zeros_like(self.conv(x))
            # computation for all postive weights
            computation = 0
            remainder = saved_w.detach() / basis
            #print('remainder',remainder[0,0,:,:])
            for i in range(bits):
                # mask for negative weights, bit by bit
                mask_wn = torch.round(remainder / 2**(bits-1-i)) 
                remainder = remainder - mask_wn * (2**(bits-1-i))
                # mask for output becomes featuremap negative
                if i==0:
                    mask_on = torch.ones_like(conv)
                else:
                    mask_on = (conv > 0).float()
                self.conv.weight.data = basis * mask_wn * (2**(bits-1-i))
                conv += self.conv(x) * mask_on
                # computation do not count if output featuremap has already been negative
                # consider sparcity of weights
                #computation += torch.sum(mask_wn*(2*mask_pos-1)) / saved_w.numel() * torch.sum(mask_on) / conv.numel() / bits.float()
                # exclude sparcity of weights
                computation += torch.sum(mask_on) / conv.numel() / bits.float()
            x = self.relu(conv)
            self.conv.weight.data = saved_w

            #false_neg = torch.sum(ground_relu[x==0] >0).float()/ x.numel()
            #print(false_neg)
            if torch.max(x-ground_relu) > 1e-2:
                print("different result of relu")

            return x, computation

        else:
            raise ValueError("early_predict mode has no {} option".format(self.early_predict))

class convrelubn_block(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=1,
            stride = 1, relu = True, early_predict=0, bn_adjust=0):
        super(convrelubn_block, self).__init__()
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

        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(self.out_planes)
    
    def forward(self, x,epoch=0):
        computation = 0

        return(self.bn(self.relu(self.conv(x)))), computation
