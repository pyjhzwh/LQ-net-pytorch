#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from ._conv_block import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.conv1 = conv3x3(inplanes, planes, stride)

        #self.bn2 = nn.BatchNorm2d(planes)
        #self.relu2 = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)

        self.conv1 = convbnrelu_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, relu=True)
        self.conv2 = convbnresrelu_block(planes, planes, kernel_size=3, padding=1, stride=1, relu=True)
        

        self.downsample = downsample

        self.stride = stride

        return

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        #print('residual size:',residual.size())

        out,_ = self.conv1(x)
        # relu(bn(conv(out)) + residual)
        out,_ = self.conv2(out, residual)
        #print('out size:',out.size())
        #out += residual

        return out

class _make_layer(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, stride =1, early_predict=0):
        super(_make_layer,self).__init__()
        self.inplanes=inplanes
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print("downsample, stride = ",stride)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        self.layers = nn.ModuleList()
        self.layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            self.layers.append(block(self.inplanes, planes))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, nclass=1000, zero_init_residual=False):
        super(ResNet,self).__init__()
        self.nclass = nclass
        self.inplanes = 64
        #self.conv1 = conv3x3(3,self.inplanes)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        self.layer0 = convbnrelu_block(3, self.inplanes,kernel_size=7, stride=2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(block,64,64,layers[0],stride=1)
        self.layer2 = _make_layer(block,64,128,layers[1],stride=2)
        self.layer3 = _make_layer(block,128,256,layers[2],stride=2)
        self.layer4 = _make_layer(block,256,512,layers[3],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512* block.expansion, nclass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    '''
    def _make_layer(self, block, planes, blocks, stride =1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print("downsample, stride = ",stride)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    '''

    def forward(self,x):

        x,_ = self.layer0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x


#def resnet20(nclass=10):
#    return ResNet(BasicBlock, [3,3,3], nclass=nclass)

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=progress)
        
        new_key_dict1={"conv1":"layer0.conv", "bn1": "layer0.bn"}
        new_key_dict2={"conv1":"conv1.conv", "bn1": "conv1.bn","conv2":"conv2.conv", "bn2":"conv2.bn"}
        for key in list(state_dict.keys()):
            #print(key)
            split_keys = key.split(".")
            if split_keys[0] in new_key_dict1.keys():
                new_key = new_key_dict1[split_keys[0]]+"."+key.split(".",1)[1]
                state_dict[new_key] = state_dict.pop(key)
            elif "layer" in split_keys[0]:
                if split_keys[2] in new_key_dict2.keys():
                    new_key = key.replace(split_keys[2], new_key_dict2[split_keys[2]])
                    new_key_split = new_key.split(".",1)
                    new_key = new_key_split[0] + ".layers." + new_key_split[1]
                    state_dict[new_key] = state_dict.pop(key)
                elif "downsample" in split_keys:
                    new_key_split = key.split(".",1)
                    new_key = new_key_split[0] + ".layers." + new_key_split[1]
                    state_dict[new_key] = state_dict.pop(key)


        model.load_state_dict(state_dict)

    return model

