#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

class fc_block(nn.Module):

    def __init__(self, in_features, out_features):
        super(fc_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x,epoch=0):
        return self.relu(self.fc(x))