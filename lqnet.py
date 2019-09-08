from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from math import *
import numpy as np

layerdict = ['module.conv1.0.weight', 'module.conv2.0.weight', 'module.conv3.0.weight','module.conv4.0.weight','module.conv5.0.weight','module.conv6.0.weight','module.conv7.0.weight']

class learned_quant():

    def __init__ (self, model, b, moving_aver=0.9):
        super(learned_quant, self).__init__()
        
        self.model = model
        self.b = b
        self.moving_aver=moving_aver

        self.W = []
        self.preW = []
        #self.testpreW = []
        self.B = []

        for key, value in model.named_parameters():
            if key in layerdict:
                print(key)
                self.W.append(value)
                self.preW.append(value.data.clone())
                #self.testpreW.append(value.data.clone())
                self.B.append(value.data.clone().zero_())

        # v is the learnable floating point basis
        self.v = torch.zeros(len(self.W),dtype=torch.float)

        for i in range(len(self.W)):
            self.v[i] = 0.5 / (pow(2,b[i])-1)


    def update(self):
        with torch.no_grad():
            for i in range(len(self.W)):
                # compute Bi with v
                self.B[i] = self.W[i].data / self.v[i]
                self.B[i] = torch.round((self.B[i]-1)/2)*2-1
                self.B[i] = torch.clamp(self.B[i], -(pow(2,self.b[i])-1), pow(2,self.b[i])-1)

                # compute v[i] with B[i]
                vi = torch.sum(torch.mul(self.B[i],self.W[i].data) / torch.sum(torch.mul(self.B[i],self.B[i])))
                # update v[i] with moving average
                self.v[i] = self.v[i] * self.moving_aver + vi * (1-self.moving_aver)

                # apply B[i] to W[i]
                self.preW[i] = self.W[i].data.clone()
                self.W[i].data = (self.B[i]*self.v[i]).clone()

    # use gradient of Bi to update Wi
    # turn W to floating point representation 
    '''
    def loss_grad(self):
        for i in range(len(self.W)):
            self.W[i].data = self.preW[i].clone()
    '''
    def apply_quantval(self):
        
        for i in range(len(self.W)):
            self.W[i] = (self.B[i] * self.v[i]).clone()

    def storequntW(self):
    
        i = 0
        for key, value in self.model.named_parameters():
            if key in layerdict:
                self.model.state_dict()[key].copy_(self.W[i])
                i = i + 1

    def restoreW(self):
        '''
        '''
        for i in range(len(self.W)):
            self.W[i].data = self.preW[i].clone()

    '''
    def teststoreW(self):
        for i in range(len(self.W)):
            self.testpreW[i] = self.W[i].data.clone()

    def testrestoreW(self):
        for i in range(len(self.W)):
            self.W[i].data = self.testpreW[i].clone()
    '''
    def print_info(self):
        print('\n' + '-' * 30)
        i = 0
        for key, value in self.model.named_parameters():
            if key in layerdict:
                print(key)
                print('W val:',self.W[i].data.view(1,-1))
                print('quant val:',(self.B[i]*self.v[i]).view(1,-1))
                print('v val:', self.v[i])
                #print('preW val:',self.preW[i].data.view(1,-1))
                print('L2 norm of W and B:',torch.norm((self.W[i]-self.B[i]*self.v[i]).view(1,-1)))
                #print('L2 norm of W and preW:',torch.norm((self.W[i]-self.preW[i]).view(1,-1)))
                i = i+1
        print('\n' + '-' * 30)

    def print_weights(self):
        print('\n' + '-' * 30)
        i = 0
        for key, value in self.model.named_parameters():
            if key in layerdict:
                print(key)
                print('W val:',self.W[i].data.view(1,-1))
                i = i+1
        print('\n' + '-' * 30)
    
        
