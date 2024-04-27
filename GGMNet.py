import os.path
import torch
import torch.nn as nn
import cv2
import numpy as np
from nets.resnet import resnet50
from nets.vgg import VGG16
import torch.nn.functional as F
import math


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class GGMNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg', ):
        super(GGMNet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [128, 256, 512, 1024]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [32, 64, 128, 256, 512, 1024]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.ca = ca(10)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None


        self.final2 = nn.Conv2d(10, 2, kernel_size=1)
        self.final3 = nn.Conv2d(32, 2, kernel_size=1)

        self.backbone = backbone
        self.co1 = conv1x1(512, 2)
        self.co2 = conv1x1(256, 2)
        self.co3 = conv1x1(128, 2)
        self.co4 = conv1x1(64, 2)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        # feat6 = self.aspp(feat5)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        a = self.co1(feat5)
        b = self.co2(up4)
        c = self.co3(up3)
        d = self.co4(up2)
        a = self.up1(a)
        b = self.up2(b)
        c = self.up3(c)
        d = self.up4(d)
        e = self.final3(up1)
        final = torch.cat([a, b, c, d, e], dim=1)
        final = self.ca(final)
        final = self.final2(final)
        return final,a,b,c,d,e

class ca(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ca, self).__init__() # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False) # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size() # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):

        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = 64

        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 128

        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 256

        self.layer4 = self._make_layer(block, 512, layers[3])

        self.maxpool = nn.MaxPool2d(2)

        self.gppm = PyramidPooling(512,512)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if self.inplanes != 0:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        layers.append(GlobalContextBlock(planes * block.expansion))
        for i in range(1, blocks):
            layers.append(block(planes, planes))
            layers.append(GlobalContextBlock(planes * block.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat1 = self.layer1(x)
        f1 = self.maxpool(feat1)
        feat2 = self.layer2(f1)
        f2 = self.maxpool(feat2)
        feat3 = self.layer3(f2)
        f3 = self.maxpool(feat3)
        feat4 = self.layer4(f3)
        feat5 = self.maxpool(feat4)
        feat5 = self.gppm(feat5)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x).permute(0, 2, 1)
        return self.conv1d_2(h).permute(0, 2, 1)


class GloRe(nn.Module):
    def __init__(self, in_channels, mid_channels, N):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.N = N

        self.phi = nn.Conv2d(in_channels, mid_channels, 1)
        self.theta = nn.Conv2d(in_channels, N, 1)
        self.gcn = GCN(N, mid_channels)
        self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        mid_channels = self.mid_channels
        N = self.N

        B = self.theta(x).view(batch_size, N, -1)
        x_reduced = self.phi(x).view(batch_size, mid_channels, h * w)
        x_reduced = x_reduced.permute(0, 2, 1)
        v = B.bmm(x_reduced)

        z = self.gcn(v)
        y = B.permute(0, 2, 1).bmm(z).permute(0, 2, 1)
        y = y.view(batch_size, mid_channels, h, w)
        x_res = self.phi_inv(y)

        return x + x_res

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = nn.Conv2d(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = nn.Conv2d(in_channels, inter_channels, 1, **kwargs)
        self.out = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.gnn = GloRe(in_channels, inter_channels,5)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]

        feat2 = self.upsample(self.relu(self.bn(self.conv1(self.pool(x, 2)))), size)
        feat2 = self.gnn(feat2)
        feat3 = self.upsample(self.relu(self.bn(self.conv1(self.pool(x, 4)))), size)
        feat3 = self.gnn(feat3)
        feat4 = self.upsample(self.relu(self.bn(self.conv1(self.pool(x, 8)))), size)
        feat4 = self.gnn(feat4)

        x = torch.cat([self.gnn(x), feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x