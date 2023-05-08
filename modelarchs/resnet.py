#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from typing import Type, Any, Callable, Union, List, Optional

from ._conv_block import *
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet20': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20.th',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

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

        self.conv1 = convbnrelu_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, relu=nn.ReLU)
        self.conv2 = convbnresrelu_block(planes, planes, kernel_size=3, padding=1, stride=1, relu=nn.ReLU)
        

        self.downsample = downsample

        self.stride = stride

        return

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        #print('residual size:',residual.size())

        out = self.conv1(x)
        # relu(bn(conv(out)) + residual)
        out = self.conv2(out, residual)
        #print('out size:',out.size())
        #out += residual

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        #self.conv1 = conv1x1(inplanes, width)
        #self.bn1 = norm_layer(width)
        #self.conv2 = conv3x3(width, width, stride, groups, dilation)
        #self.bn2 = norm_layer(width)
        #self.conv3 = conv1x1(width, planes * self.expansion)
        #self.bn3 = norm_layer(planes * self.expansion)
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = convbnrelu_block(inplanes, width, kernel_size=1, padding=0, stride=1, relu=nn.ReLU)
        self.conv2 = convbnrelu_block(width, width, kernel_size=3, padding=1, stride=stride, relu=nn.ReLU)
        self.conv3 = convbnresrelu_block(width, planes * self.expansion, kernel_size=1, padding=0, stride=1, relu=nn.ReLU)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x) :
        identity = x

        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        #out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        #out = self.relu(out)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out, identity)

        return out

class _make_layer(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, stride =1, option='A'):
        super(_make_layer,self).__init__()
        self.inplanes=inplanes
        downsample = None
        if option == 'A':
            '''
            imagenet
            '''
            if stride != 1 or self.inplanes != planes * block.expansion:
                #print("downsample, stride = ",stride)
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion)
                )
        elif option == 'B':
            '''
            cifar-10
            '''
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

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
    def __init__(self, name, block, layers, nclass=1000, zero_init_residual=False, expansion=1):
        super(ResNet,self).__init__()
        self.name = name
        self.nclass = nclass
        self.inplanes = 64
        self.expansion = expansion
        #self.conv1 = conv3x3(3,self.inplanes)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        self.layer0 = convbnrelu_block(3, self.inplanes,kernel_size=7, stride=2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(block,64,64,layers[0],stride=1)
        self.layer2 = _make_layer(block,64*block.expansion,128,layers[1],stride=2)
        self.layer3 = _make_layer(block,128*block.expansion,256,layers[2],stride=2)
        self.layer4 = _make_layer(block,256*block.expansion,512,layers[3],stride=2)

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


    def forward(self,x):

        x = self.layer0(x)
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
    model = ResNet("resnet18", BasicBlock, [2, 2, 2, 2], **kwargs)
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



def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model =  ResNet("resnet50", Bottleneck, [3, 4, 6, 3], expansion=4, **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=progress)
        
        new_key_dict1={"conv1":"layer0.conv", "bn1": "layer0.bn"}
        new_key_dict2={"conv1":"conv1.conv", "bn1": "conv1.bn","conv2":"conv2.conv", "bn2":"conv2.bn",
                    "conv3":"conv3.conv", "bn3": "conv3.bn"}
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


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet("resnet101", Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101'],
                                              progress=progress)
        new_key_dict1={"conv1":"layer0.conv", "bn1": "layer0.bn"}
        new_key_dict2={"conv1":"conv1.conv", "bn1": "conv1.bn","conv2":"conv2.conv", "bn2":"conv2.bn",
                    "conv3":"conv3.conv", "bn3": "conv3.bn"}
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

def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet("resnet152", Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152'],
                                              progress=progress)
        new_key_dict1={"conv1":"layer0.conv", "bn1": "layer0.bn"}
        new_key_dict2={"conv1":"conv1.conv", "bn1": "conv1.bn","conv2":"conv2.conv", "bn2":"conv2.bn",
                    "conv3":"conv3.conv", "bn3": "conv3.bn"}
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

class ResNet_cifar10(nn.Module):
    def __init__(self, block, layers, nclass=10, zero_init_residual=False):
        super(ResNet_cifar10,self).__init__()
        self.nclass = nclass
        self.inplanes = 16
        #self.conv1 = conv3x3(3,self.inplanes)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        self.layer0 = convbnrelu_block(3, self.inplanes,kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,16,layers[0],stride=1, option='B')
        self.layer2 = self._make_layer(block,32,layers[1],stride=2, option='B')
        self.layer3 = self._make_layer(block,64,layers[2],stride=2, option='B')

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64* block.expansion, nclass)

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


    def forward(self,x):

        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.avgpool(x)
        x = F.avg_pool2d(x, x.shape[3])
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x

def resnet20(pretrained: bool = True, progress: bool = True, **kwargs):
    r"""ResNet-20 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet_cifar10(BasicBlock, [3, 3, 3], **kwargs)
    #print('pretrained',pretrained)
    if pretrained:
        chk = torch.load('./saved_models/resnet20-12fca82f.th')
        state_dict = chk['state_dict']
        
        new_key_dict1={"conv1":"layer0.conv", "bn1": "layer0.bn"}
        new_key_dict2={"conv1":"conv1.conv", "bn1": "conv1.bn","conv2":"conv2.conv", "bn2":"conv2.bn"}
        for key in list(state_dict.keys()):
            #print(key)
            split_keys = key.split(".")
            if split_keys[1] in new_key_dict1.keys():
                new_key = new_key_dict1[split_keys[1]]+"."+split_keys[2]
                state_dict[new_key] = state_dict.pop(key)
            
            elif "layer" in split_keys[1]:
                if split_keys[3] in new_key_dict2.keys():
                    #new_key = key.replace(split_keys[2], new_key_dict2[split_keys[2]])
                    #new_key_split = new_key.split(".",1)
                    new_key = split_keys[1] + ".layers." + split_keys[2]+"."+new_key_dict2[split_keys[3]]+"."+split_keys[4]
                    state_dict[new_key] = state_dict.pop(key)
        
            elif "linear" in split_keys[1]:
                new_key = "fc."+split_keys[2]
                state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict)

    return model
