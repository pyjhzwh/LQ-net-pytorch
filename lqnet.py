from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from math import *
import numpy as np

#layerdict = ['module.conv1.0.weight', 'module.conv2.0.weight', 'module.conv3.0.weight','module.conv4.0.weight','module.conv5.0.weight','module.conv6.0.weight','module.conv7.0.weight']

NORM_PPF_0_75 = 0.6745

class learned_quant():

    def __init__ (self, target_weights, b, moving_aver=0.9, needbias=False):
        super(learned_quant, self).__init__()
        
        #self.model = model
        self.b = b
        self.moving_aver=moving_aver

        self.W = target_weights
        self.preW = []
        # self.testpreW = []
        self.B = []
        self.v = []
        self.Wmean = []
        #self.zero_point = []
        self.needbias = needbias

        for (i,weights) in enumerate(self.W):
            self.v.append(torch.max(weights.data)/ (2**self.b[i]-1))
            #self.v.append(weights.data.abs().mean()/4.0)
            #n = weights.shape[1] * weights.shape[2] * weights.shape[3]
            #self.v.append(NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (self.b[i] -1)))
            self.preW.append(weights.data.clone())
            self.B.append(weights.data.clone().zero_())
            # layer-wise mean of target weight
            if self.needbias:
                self.Wmean.append(torch.mean(weights.data))
            else:
                self.Wmean.append(0)
            #self.zero_point.append(0)

        '''
        i = 0
        for key, value in model.named_parameters():
            if key in layerdict:
                print(key)
                n = value.size()[1] * value.size()[2] * value.size()[3]
                self.W.append(value)
                self.preW.append(value.data.clone())
                #self.testpreW.append(value.data.clone())
                self.B.append(value.data.clone().zero_())
                self.v[i] = value.abs().mean() / 2.0 #NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (self.b[i] - 1))
                i = i + 1
        '''
        # v is the learnable floating point basis



    def update(self, test=False):
        for i in range(len(self.W)):
            # compute Bi with v
            self.B[i] = (self.W[i].data - self.Wmean[i]) / self.v[i]
            # now Q = round(float / scale + zero_point)
            # scale: float, zero_point:int, odd
            #self.zero_point[i] = torch.round( (-self.Wmean[i] / self.v[i] -1)/2)*2 + 1
            #self.B[i] = self.W[i].data / self.v[i] + self.zero_point[i]
            self.B[i] = torch.round((self.B[i]-1)/2)*2+1
            self.B[i] = torch.clamp(self.B[i], -(pow(2,self.b[i])-1), pow(2,self.b[i])-1)

            # compute v[i] with B[i]
            vi = torch.sum(torch.mul(self.B[i],(self.W[i].data - self.Wmean[i])) / torch.sum(torch.mul(self.B[i],self.B[i])))
            # update v[i] with moving average
            if not test:
                self.v[i] = self.v[i] * self.moving_aver + vi * (1-self.moving_aver)

            # apply B[i] to W[i]
            #self.preW[i].copy_(self.W[i].data)
            self.W[i].data.copy_(self.B[i]*self.v[i] + self.Wmean[i])
            # if test:
            #     print("apply W", self.W[i].data[0,0,0,:].view(1,-1))
            # #self.W[i].data.copy_((self.B[i] - self.zero_point[i]) * self.v[i])
            # update Wmean
            if self.needbias:
                self.Wmean[i] = torch.mean(self.W[i].data)

    # use gradient of Bi to update Wi
    # turn W to floating point representation 
    '''
    def loss_grad(self):
        for i in range(len(self.W)):
            self.W[i].data = self.preW[i].clone()
    '''
    def restoreW(self):
        for i in range(len(self.W)):
            self.W[i].data.copy_(self.preW[i])

    
    def storeW(self):
        for i in range(len(self.W)):
            self.preW[i].copy_(self.W[i].data)


    def apply(self, test=False):
        self.storeW()
        self.update(test=test)
        return

    def apply_quantval(self):
        
        for i in range(len(self.W)):
            self.W[i].data.copy_(self.B[i] * self.v[i]+self.Wmean[i])
            #self.W[i].data.copy_((self.B[i] - self.zero_point[i]) * self.v[i])

    def storequntW(self):
    
        i = 0
        for key, value in self.model.named_parameters():
            if key in layerdict:
                self.model.state_dict()[key].copy_(self.B[i]*self.v[i]+self.Wmean[i])
                #self.model.state_dict()[key].copy_((self.B[i] - self.zero_point[i])* self.v[i])
                i = i + 1

    '''
    def testrestoreW(self):
        for i in range(len(self.W)):
            self.W[i].data = self.testpreW[i].clone()
    '''
    def print_info(self):
        print('\n' + '-' * 30)
        #for key, value in self.model.named_parameters():
            #if key in layerdict:
        for i in range(len(self.W)):
            print(i,' layer')
            print('W size:', self.W[i].data.size())
            print('W[0,0,0,:] val:',self.W[i][0,0,0,:].data.view(1,-1))
            print('quant val:',(self.B[i]*self.v[i]+self.Wmean[i])[0].view(1,-1))
            #print('quant val:',((self.B[i] + torch.round(self.Wmean[i] / self.v[i]))* self.v[i]).view(1,-1))
            print('v val:', self.v[i])
            print('bias val:',self.Wmean[i])
            #print('zero_point val', self.zero_point[i])
            #print('bits utlized:', torch.log2(torch.max(self.B[i].abs()+1)))
            #print('preW val:',self.preW[i].data.view(1,-1))
            #print('L2 norm of W and B:',torch.norm((self.W[i]-self.B[i]*self.v[i]).view(1,-1)))
            #print('L2 norm of W and preW:',torch.norm((self.W[i]-self.preW[i]).view(1,-1)))
        print('\n' + '-' * 30)

    def save_quantinfo(self):
        quant_info=[]
        i = 0
        for i in range(len(self.W)):
            quant_info_i = {}
            quant_info_i['bias'] = self.Wmean[i]
            quant_info_i['scale'] = self.v[i]
            quant_info.append(quant_info_i)

        return quant_info
    


'''
LQ-net for quantizating activations
'''
class lq_act():

    def __init__(self, key, b, moving_aver=0.9):
        super(lq_act, self).__init__()

        self.b = b
        self.moving_aver=moving_aver
        self.key = key
        self.mean = 0
        self.err = 0
        self.prevx = None

    '''
    def storex(self, x):
        self.prevx.copy_(x.data)
    
    def restorex(self):
        return self.prevx

    def apply(self, x, stats, test=False):
        self.prevx.copy_(x.data)
        self.update(x, stats, test=test)
        x.data.copy_(self.qx)
    '''

    def update(self, x, stats, test=False):

        with torch.no_grad():
        
            if self.key not in stats:
                # init 
                self.v = torch.max(x.data) / (2**self.b-1)
                self.xmean = torch.mean(x)
                self.zero_point = -1 * torch.round(self.xmean / self.v)
                
                self.B = x.data / self.v + self.zero_point
                self.B = torch.round((self.B-1)/2)*2+1
                self.B = torch.clamp(self.B, -(pow(2,self.b)-1), pow(2,self.b)-1)
                
                stats[self.key] = {"v": self.v, "zero_point":self.zero_point, "bias":self.xmean}

                qx = (self.B - self.zero_point) * self.v

            else:
                self.B = x.data / self.v + self.zero_point
                self.B = torch.round((self.B-1)/2)*2+1
                self.B = torch.clamp(self.B, -(pow(2,self.b)-1), pow(2,self.b)-1)

                vi = torch.sum(torch.mul(self.B,(x.data + self.zero_point*self.v)) / torch.sum(torch.mul(self.B,self.B)))
                # update v with moving average
                if not test:
                    self.v = self.v * self.moving_aver + vi * (1-self.moving_aver)

                qx = (self.B - self.zero_point) * self.v
                # update xmean
                self.xmean = torch.mean(x.data)
                self.zero_point = -1 * torch.round(self.xmean / self.v)
                # update stats
                stats[self.key] = {"v": self.v,  "zero_point":self.zero_point, "bias": self.xmean}

            if not test:
                self.err = self.err + torch.norm(x - qx)/x.numel() * self.moving_aver


        return qx

    def print_info(self):
        print('\n' + '-' * 30)
        print('layer ', self.key)
        print('v val:', self.v)
        print('zero_point val:', self.zero_point)
        print('bias val:',self.xmean)
        print('quant err norm:', self.err)

