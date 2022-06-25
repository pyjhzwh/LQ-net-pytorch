from typing import Callable, Any, Optional, List

import torch
from torch import Tensor
from torch import nn

from ._conv_block import convbnrelu_block
from torchvision.ops.misc import Conv2dNormActivation as Conv2dNormActivation
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights as MobileNet_V2_Weights

__all__ = ["MobileNetV2", "mobilenet_v2"]


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                convbnrelu_block(inp, hidden_dim, kernel_size=1, relu=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                convbnrelu_block(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding = 1,
                    stride=stride,
                    groups=hidden_dim,
                    relu=nn.ReLU6,
                ),
                # pw-linear
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
                convbnrelu_block(
                    hidden_dim, oup, kernel_size=1, stride=1, padding=0,
                    usebn=True, relu=None,
                ),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            convbnrelu_block(
                input_channel, self.last_channel, kernel_size=1, relu=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# MobileNet_V2_Weights.IMAGENET1K_V1
# "ImageNet-1K": {
#     "acc@1": 71.878,
#     "acc@5": 90.286,
# }
def mobilenet_v2(
    *, weights: Optional[MobileNet_V2_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV2:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.
    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """
    weights = MobileNet_V2_Weights.verify(weights)

    # if weights is not None:
    #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV2(**kwargs)

    if weights is not None:
        # model.load_state_dict(weights.get_state_dict(progress=progress))
        state_dict = weights.get_state_dict(progress=progress)
        # print("keys", state_dict.keys())
        map_dict={"0":"conv", "1":"bn"}
        map_dict2={"1":"conv", "2":"bn"}
        map_dict3={"2":"conv", "3":"bn"}
        two_layers=["1"]
        # three_layers=["2","3","4","5","7","8","9","10","11","12","13"]
        for key in list(state_dict.keys()):
            split_keys = key.split(".")
            if split_keys[1] == "0":
                # print(f"key={key}")
                continue
            elif split_keys[1] == "18":
                new_key = f"{'.'.join(split_keys[:2])}.{map_dict[split_keys[2]]}.{'.'.join(split_keys[3:])}"
                state_dict[new_key] = state_dict.pop(key)
            elif len(split_keys) > 4:
                if split_keys[4] in map_dict.keys():
                    new_key =f"{'.'.join(split_keys[:4])}.{map_dict[split_keys[4]]}.{'.'.join(split_keys[5:])}"
                elif split_keys[3].isnumeric():
                    if split_keys[1] in two_layers:
                        layer_id = str(int(split_keys[3])// 3 + 1)
                        new_key = f"{'.'.join(split_keys[:3])}.{layer_id}.{map_dict2[split_keys[3]]}.{'.'.join(split_keys[4:])}"
                    else:
                        layer_id = str(int(split_keys[3])// 2 + 1)
                        new_key = f"{'.'.join(split_keys[:3])}.{layer_id}.{map_dict3[split_keys[3]]}.{'.'.join(split_keys[4:])}"
                        
                    # and split_keys[2] in map_dict.keys():
                    # new_key =".".join(split_keys[:2]) + map_dict[split_keys[2]] + ".".join(split_keys[3:])
                    # state_dict[new_key] = state_dict.pop(key)
                
                state_dict[new_key] = state_dict.pop(key)
                # print(f"key={key}, new_key={new_key}")
                # key=features.1.conv.0.0.weight, new_key=features.1.conv.0.conv.weight
                # exit()
        # print("new keys", state_dict.keys())
        model.load_state_dict(state_dict)

    return model