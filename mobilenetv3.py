"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch
import torch.nn as nn
import torch.nn.functional as nnf

import math


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode,  width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers

        # self.conv = conv_1x1_bn(input_channel, 512*49)
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.conv_adj = nn.Conv2d(in_channels=40*6, out_channels=512*49, kernel_size=1) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output_channel = {'large': 512, 'small': 512}
        # output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     h_swish(),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )
        # self.linear = nn.Linear(exp_size, output_channel)
        # self.outMobiNet = nnf.interpolate(self.linear, size=(7, 7), mode='bicubic', align_corners=False)
        # load_state_dict(torch.load("/home/dmp/1.User/huy.hhoang/1/distill/Distillation_ViVQA/pretrained_mobileNet/mobilenetv3-small-55df8e1f.pth"),strict=False)
        self._initialize_weights()

    def forward(self, x):
        # print("!!!!!!!!!!!!!!!!")
        # print(x)
        x = x['pixel_values']
        x = self.features(x)
        x = self.conv(x)
        # print("VIT:",x.shape)
        x = self.conv_adj(x)
        # print("VIT:",x.shape)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        # print("VIT:",x.shape)
        x = x.view(-1,49,512)
        # print(x.shape)

        

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV3ViT(nn.Module):
    def __init__(self, cfgs, mode,  width_mult=1.):
        super(MobileNetV3ViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers

        # self.conv = conv_1x1_bn(input_channel, 197*48)
        # self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # self.conv = conv_1x1_bn(input_channel, 768*197)
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.conv_adj = nn.Conv2d(in_channels=40*6, out_channels=768*197, kernel_size=1) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # output_channel = {'large': 512, 'small': 512}
        # output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     h_swish(),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )
        # self.linear = nn.Linear(exp_size, output_channel)
        # self.outMobiNet = nnf.interpolate(self.linear, size=(7, 7), mode='bicubic', align_corners=False)
        self._initialize_weights()

    def forward(self, x):
        # print("!!!!!!!!!!!!!!!!")
        # print(x)
        x = x['pixel_values']
        x = self.features(x)
        x = self.conv(x)
        # print("VIT:",x.shape)
        x = self.conv_adj(x)
        # print("VIT:",x.shape)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        # print("VIT:",x.shape)
        x = x.view(-1,197,768)
        
        

        return x
        # x = x.view(x.size(0), -1)
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    # cfgs = [
    #     # k, t, c, SE, HS, s 
    #     [3,    1,  16, 1, 0, 2],
    #     [3,  4.5,  24, 0, 0, 2],
    #     [3, 3.67,  24, 0, 0, 1],
    #     [5,    4,  40, 1, 1, 2],
    #     [5,    6,  40, 1, 1, 1],
    #     [5,    6,  40, 1, 1, 1],
    #     [5,    6,  512, 1, 1, 1],
    #     [5,    6,  1, 1, 1, 1],
    # ]
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, mode='small', **kwargs)

def mobilenetv3_smallViT(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    # cfgs = [
    #     # k, t, c, SE, HS, s 
    #     [3,    1,  16, 1, 0, 2],
    #     [3,  4.5,  24, 0, 0, 2],
    #     [3, 3.67,  24, 0, 0, 1],
    #     [5,    4,  40, 1, 1, 2],
    #     [5,    6,  40, 1, 1, 1],
    #     [5,    6,  40, 1, 1, 1],
    #     [5,    6,  768, 1, 1, 1],
    #     [5,    6,  1, 1, 1, 1],
    # ]
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
    ]

    return MobileNetV3ViT(cfgs, mode='small', **kwargs)
