#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

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




class conv_block(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, padding=1,
            stride = 1, relu = True, early_predict=0):
        super(conv_block, self).__init__()
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

        self.conv = nn.Conv2d(self.in_planes, self.out_planes, self.kernel_size,
                padding=self.padding, stride=self.stride, bias = False)
        
        self.bn = nn.BatchNorm2d(self.out_planes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        computation = 0

        if self.early_predict == 0:
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


class all_cnn_net(nn.Module):
    def __init__(self, nclass=10, early_predict=0):
        super(all_cnn_net,self).__init__()
        self.nclass = nclass
        self.early_predict = early_predict

        self.conv0 = conv_block(3, 96, kernel_size=3, padding=1, stride=1, relu=True,early_predict=0)
        self.conv1 = conv_block(96, 96, kernel_size=3, padding=1, stride=1, relu=True,early_predict=self.early_predict)
        self.conv2 = conv_block(96, 96, kernel_size=3, padding=1, stride=2, relu=True,early_predict=self.early_predict)

        self.dropout0 = nn.Dropout(p=0.5)

        self.conv3 = conv_block(96, 192, kernel_size=3, padding=1, stride=1, relu=True,early_predict=self.early_predict)
        self.conv4 = conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True,early_predict=self.early_predict)
        self.conv5 = conv_block(192, 192, kernel_size=3, padding=1, stride=2, relu=True,early_predict=self.early_predict)

        self.dropout1 = nn.Dropout(p=0.5)

        self.conv6 = conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True,early_predict=self.early_predict)
        self.conv7 = conv_block(192, 192, kernel_size=1, padding=0, stride=1, relu=True,early_predict=self.early_predict)
        self.conv8 = conv_block(192, 10, kernel_size=1, padding=0, stride=1, relu=True,early_predict=0)

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


    def forward(self,x):

        computation = torch.zeros(7, dtype=torch.float).cuda()
        x,_ = self.conv0(x)
        x,computation[0] = self.conv1(x) 
        x,computation[1] = self.conv2(x)

        x = self.dropout0(x)

        x,computation[2] = self.conv3(x)
        x,computation[3] = self.conv4(x)
        x,computation[4] = self.conv5(x)

        x = self.dropout1(x)

        x,computation[5] = self.conv6(x)
        x,computation[6] = self.conv7(x)
        x,_ = self.conv8(x)

        #x = self.avgpool(x)
        x = F.avg_pool2d(x, kernel_size=x.size(2))
        x = x.view(x.size(0),-1)
        
        return x, computation
    
    def layer_computation_weight(self,x):

        computation_weight = torch.zeros(7, dtype=torch.float).cuda()
        x,_ = self.conv0(x)
        x,_ = self.conv1(x)
        computation_weight[0] = (self.conv1.conv.weight.data.shape[1] * self.conv1.conv.weight.data.shape[2] * self.conv1.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])
        x,_ = self.conv2(x)
        computation_weight[1] = (self.conv2.conv.weight.data.shape[1] * self.conv2.conv.weight.data.shape[2] * self.conv2.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])
        
        x = self.dropout0(x)

        x,_ = self.conv3(x)
        computation_weight[2] = (self.conv3.conv.weight.data.shape[1] * self.conv3.conv.weight.data.shape[2] * self.conv3.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])
        x,_ = self.conv4(x)
        computation_weight[3] = (self.conv4.conv.weight.data.shape[1] * self.conv4.conv.weight.data.shape[2] * self.conv4.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])
        x,_ = self.conv5(x)
        computation_weight[4] = (self.conv5.conv.weight.data.shape[1] * self.conv5.conv.weight.data.shape[2] * self.conv5.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])

        x = self.dropout1(x)

        x,_ = self.conv6(x)
        computation_weight[5] = (self.conv6.conv.weight.data.shape[1] * self.conv6.conv.weight.data.shape[2] * self.conv6.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])
        x,_ = self.conv7(x)
        computation_weight[6] = (self.conv7.conv.weight.data.shape[1] * self.conv7.conv.weight.data.shape[2] * self.conv7.conv.weight.data.shape[3]) **2 * (x.shape[1] * x.shape[2] * x.shape[3])


        return computation_weight

    '''
    def _conv_block(self, in_planes, out_planes, kernel_size=1, padding=1,
            stride = 1, relu = True):
        if relu:
            conv_block = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False),
                    nn.ReLU(inplace = True)
                    )
        else:
            raise Exception ('Currently all blocks use ReLU')
        return conv_block
    

    def forward(self, x, computation):

        # conv0
        x = self.conv0(x)
        x = self.relu0(x)

        # groudtruth conv1
        ground_conv1 = self.conv1(x)
        ground_relu1 = self.relu1(x)
        # conv1
        saved_w = self.conv1.weight.data
        mask_pos = (saved_w >= 0).float()
        self.conv1.weight.data = saved_w * mask_pos
        conv1 = self.conv1(x)
        # computation for all postive weights
        computation[1] = self.bits[1] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() * (1-mask_pos) / self.basis[1]
        print("saved_w", saved_w)
        print("self.basis[1]", self.basis[1])
        for i in range(self.bits[1]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[1]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[1]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv1 > 0).float()
            self.conv1.weight.data = self.basis[1] * mask_wn * (2**(self.bits[1] -i -1))
            conv1 += self.conv1(x) * mask_on
            # computation do not count if output featuremap has already been negative
            computation[1] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on) / conv1.numel()
        computation[1] /= self.bits[1] 
        print('computation[1]',computation[1])
        self.conv1.weight.data = saved_w
        x = self.relu1(conv1)

        if torch.all(torch.eq(x,ground_relu1)):
            print("same result of relu1")
        else:
            print("different result of relu1")


        # conv2
        saved_w = self.conv2.weight.data
        mask_pos = (saved_w >= 0).float()
        self.conv2.weight.data = saved_w * mask_pos * mask
        conv2 = self.conv2(x)
        computation[2] = self.bits[2] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() * (1-mask_pos) / self.basis[2]
        for i in range(self.bits[2]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[2]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[2]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv2 > 0).float()
            self.conv2.weight.data = saved_w * mask_wn
            conv2 += self.conv2(x) * mask_on
            computation[2] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on)/ conv2.numel()
        computation[2] /= self.bits[2]
        #print('computation[2]',computation[2])
        self.conv2.weight.data = saved_w
        x = self.relu2(conv2)

        # dropout0
        x = self.dropout0(x)

        # conv3
        saved_w = self.conv3.weight.data
        mask_pos = (saved_w > 0).float()
        self.conv3.weight.data = saved_w * mask_pos
        conv3 = self.conv3(x)
        computation[3] = self.bits[3] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() * (1-mask_pos)/ self.basis[3]
        for i in range(self.bits[3]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[3]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[3]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv3 > 0).float()
            self.conv3.weight.data = saved_w * mask_wn
            conv3 += self.conv3(x) * mask_on
            computation[3] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on)/ conv3.numel()
        computation[3] /= self.bits[3]
        #print('computation[3]',computation[3])
        self.conv3.weight.data = saved_w
        x = self.relu3(conv3)
    
        # conv4
        saved_w = self.conv4.weight.data
        mask_pos = (saved_w > 0).float()
        self.conv4.weight.data = saved_w * mask_pos
        conv4 = self.conv4(x)
        computation[4] = self.bits[4] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() * (1-mask_pos)/ self.basis[4]
        for i in range(self.bits[4]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[4]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[4]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv4 > 0).float()
            self.conv4.weight.data = saved_w * mask_wn
            conv4 += self.conv4(x) * mask_on
            computation[4] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on)/ conv4.numel()
        computation[4] /= self.bits[4]
        #print('computation[4]',computation[4])
        self.conv4.weight.data = saved_w
        x = self.relu4(conv4)
    
        # conv5
        saved_w = self.conv5.weight.data
        mask_pos = (saved_w > 0).float()
        self.conv5.weight.data = saved_w * mask_pos
        conv5 = self.conv5(x)
        computation[5] = self.bits[5] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() * (1-mask_pos)/ self.basis[5]
        for i in range(self.bits[5]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[5]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[5]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv5 > 0).float()
            self.conv5.weight.data = saved_w * mask_wn
            conv5 += self.conv5(x) * mask_on
            computation[5] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on)/ conv5.numel()
        computation[5] /= self.bits[5]
        #print('computation[5]',computation[5])
        self.conv5.weight.data = saved_w
        x = self.relu5(conv5)

        # dropout1
        x = self.dropout1(x)

        # conv6
        saved_w = self.conv6.weight.data
        mask_pos = (saved_w > 0).float()
        self.conv6.weight.data = saved_w * mask_pos
        conv6 = self.conv6(x)
        computation[6] = self.bits[6] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() *(1-mask_pos)/ self.basis[6]
        for i in range(self.bits[6]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[6]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[6]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv6 > 0).float()
            self.conv6.weight.data = saved_w * mask_wn
            conv6 += self.conv6(x) * mask_on
            computation[6] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on)/ conv6.numel()
        computation[6] /= self.bits[6] 
        #print('computation[6]',computation[6])
        self.conv6.weight.data = saved_w
        x = self.relu6(conv6)

        # conv7
        saved_w = self.conv7.weight.data
        mask_pos = (saved_w > 0).float()
        self.conv7.weight.data = saved_w * mask_pos
        conv7 = self.conv7(x)
        computation[7] = self.bits[7] * torch.sum(mask_pos) / saved_w.numel()
        remainder = saved_w.detach() * (1-mask_pos)/ self.basis[7]
        for i in range(self.bits[7]):
            # mask for negative weights, bit by bit
            mask_wn = torch.floor(remainder / 2**(self.bits[7]-1-i))
            remainder = remainder - mask_wn * (2**(self.bits[7]-1-i))
            # mask for output becomes featuremap negative
            mask_on = (conv7 > 0).float()
            self.conv7.weight.data = saved_w * mask_wn
            conv7 += self.conv7(x) * mask_on
            computation[7] += (i+1) * torch.sum(mask_wn) / saved_w.numel() * torch.sum(mask_on)/ conv7.numel()
        computation[7] /= self.bits[7]
        #print('computation[7]',computation[7])
        self.conv7.weight.data = saved_w
        x = self.relu7(conv7)

        # conv8
        x = self.conv8(x)
        x = self.relu8(x)

        # avg_pool
        x = F.avg_pool2d(x, kernel_size=x.size(2))

        # reshape
        x = x.view(x.size(0),-1)

        return x

    ''' 
