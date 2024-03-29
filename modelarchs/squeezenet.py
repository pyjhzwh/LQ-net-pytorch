import torch
import torch.nn as nn
import torch.nn.init as init
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from typing import Any
from ._conv_block import *

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        quantAct=False,
        bits=None,
        key=''
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        '''
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        '''
        self.squeeze = convbnrelu_block(inplanes, squeeze_planes, 
                                    kernel_size=1, usebn=False,
                                    quantAct=quantAct, bits=bits,key=key)
        self.expand1x1 = convbnrelu_block(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, usebn=False,
                                    quantAct=quantAct, bits=bits,key=key+1)
        self.expand3x3 = convbnrelu_block(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1, usebn=False,
                                    quantAct=quantAct, bits=bits,key=key+2)

    def forward(self, x: torch.Tensor, stats=None) -> torch.Tensor:

        x = self.squeeze(x, stats)

        return torch.cat([
            self.expand1x1(x, stats),
            self.expand3x3(x, stats)
        ], 1)



class SqueezeNet(nn.Module):

    def __init__(
        self,
        version: str = '1_1',
        num_classes: int = 1000,
        quantAct = False,
        bits=8
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
            self.relu0 = nn.ReLU(inplace=True)
            self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire1 = Fire(64, 16, 64, 64, quantAct=quantAct, bits=bits, key=0)
            self.fire2 = Fire(128, 16, 64, 64, quantAct=quantAct, bits=bits, key=3)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire3 = Fire(128, 32, 128, 128, quantAct=quantAct, bits=bits, key=6)
            self.fire4 = Fire(256, 32, 128, 128, quantAct=quantAct, bits=bits, key=9)
            self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            self.fire5 = Fire(256, 48, 192, 192, quantAct=quantAct, bits=bits, key=12)
            self.fire6 = Fire(384, 48, 192, 192, quantAct=quantAct, bits=bits, key=15)
            self.fire7 = Fire(384, 64, 256, 256, quantAct=quantAct, bits=bits, key=18)
            self.fire8 = Fire(512, 64, 256, 256, quantAct=quantAct, bits=bits, key=21)
            '''
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
            '''
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.target_acts = [self.fire1, self.fire2, self.fire3, self.fire4, 
                            self.fire5, self.fire6, self.fire7, self.fire8]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, stats):
        #x = self.features(x)

        x = self.conv0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        x = self.fire1(x, stats)
        x = self.fire2(x, stats)
        x = self.pool2(x)

        x = self.fire3(x, stats)
        x = self.fire4(x, stats)
        x = self.pool4(x)

        x = self.fire5(x, stats)
        x = self.fire6(x, stats)
        x = self.fire7(x, stats)
        x = self.fire8(x, stats)

        
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def print_Actinfo(self):

        for layer in self.target_acts:
            layer.squeeze.lqAct.print_info()
            layer.expand1x1.lqAct.print_info()
            layer.expand3x3.lqAct.print_info()

def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        new_key_dict={"0":"conv0", "3":"fire1", "4":"fire2", "6":"fire3",
            "7":"fire4", "9":"fire5", "10":"fire6", "11":"fire7", "12":"fire8"}
        for key in list(state_dict.keys()):
            if 'features' in key:
                split_key = key.split(".")
                if ('squeeze' in key or 'expand' in key):
                    if 'bias' not in key:
                        new_key = new_key_dict[split_key[1]]+"."+split_key[2]+".conv."+split_key[3]
                        state_dict[new_key] = state_dict.pop(key)
                    else:
                        new_key = new_key_dict[split_key[1]]+"."+split_key[2]+".bn."+split_key[3]
                        state_dict[new_key] = state_dict.pop(key)
                elif split_key[1] == '0':
                    state_dict["conv0."+split_key[2]] = state_dict.pop(key)


        model.load_state_dict(state_dict,strict=False)
    return model


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)