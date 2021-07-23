# Changes:
# 1) Adding multi-scale detection (FPN), but only 2 layers
# 2) Increasing mask image size, from 28x28 to 112 x 112

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


class Bottleneck(nn.Module):

    def __init__(self, in_channel, base_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.expand = 4  # 64 -> 256, 128 -> 512 etc.

        self.conv1 = nn.Conv2d(in_channel, base_channel, kernel_size=1, stride=stride)  # This step will
        self.bn1 = nn.BatchNorm2d(base_channel)  # eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(base_channel)
        self.conv3 = nn.Conv2d(base_channel, base_channel * self.expand, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(base_channel * self.expand)
        self.relu = nn.ReLU()

        self.stride = stride
        self.downsample = downsample

        self.padding = SamePad2d(3, 1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        loop_times = [3, 4, 6, 3]  # for stage 2, 3, 4, 5, respectively
        self.bottleneck = Bottleneck
        self.inchannel = 64

        self.S1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(), SamePad2d(kernel_size=3, stride=2),
                                nn.MaxPool2d(kernel_size=3, stride=2))  # 224 -> 56
        self.S2 = self.make_block(64, loop_times[0], stride=1)  # 56 -> 56, because the stride is 1
        self.S3 = self.make_block(128, loop_times[1], stride=2)  # 56 -> 28
        self.S4 = self.make_block(256, loop_times[2], stride=2)  # 28 -> 14

    def forward(self, x):
        out = self.S1(x)
        out = self.S2(out)
        out = self.S3(out)
        out = self.S4(out)
        # print('out shape:', out.shape)
        return out

    def layers(self):
        return self.S1, self.S2, self.S3, self.S4

    def make_block(self, base_channel, loops, stride=1):
        downsampling = None

        # I guess if stride is bigger than 1, it will cause shrink of feature map size
        if stride > 1 or self.inchannel != 4 * base_channel:
            downsampling = nn.Sequential(
                nn.Conv2d(self.inchannel, base_channel * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(base_channel * 4)
            )

        layers = []

        # only first bottleneck will apply the customed stride
        layers.append(self.bottleneck(self.inchannel, base_channel, stride, downsampling))
        self.inchannel = base_channel * 4

        # The rest bottleneck will keep using 1 stride, make sure not decrease the size of image
        for i in range(1, loops):
            layers.append(self.bottleneck(self.inchannel, base_channel))
        return nn.Sequential(*layers)


# Feature Pyramid Net
class FPN(nn.Module):
    def __init__(self, S1, S2, S3, S4, outchannel):
        super(FPN, self).__init__()
        self.S1 = S1
        self.S2 = S2
        self.S3 = S3
        self.S4 = S4

        self.S4_conv = nn.Conv2d(1024, outchannel, kernel_size=1, stride=1)  # Only conv on this layer
        self.S3_conv = nn.Conv2d(512, outchannel, kernel_size=1, stride=1)  # Only conv on this layer
        self.S2_conv = nn.Conv2d(256, outchannel, kernel_size=1, stride=1)  # Only conv on this layer

        self.conv_both = nn.Sequential(  # Conv on added layer
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1)
        )

    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        S2_x = x
        x = self.S3(x)
        S3_x = x
        x = self.S4(x)

        S4_out = self.S4_conv(x)
        S3_out = self.S3_conv(S3_x) + F.upsample(S4_out, scale_factor=2)
        S2_out = self.S2_conv(S2_x) + F.upsample(S3_out, scale_factor=2)

        S4_out = self.conv_both(S4_out)
        S3_out = self.conv_both(S3_out)
        S2_out = self.conv_both(S2_out)

        out = torch.cat([F.interpolate(S2_out, scale_factor=0.5), S3_out, F.upsample(S4_out, scale_factor=2)], dim=1)
        # print('S2', S2_out.shape)
        # print('S3', S3_out.shape)
        # print('out.shape', out.shape)
        return out  # [S2_out, S3_out]


# Mask Prediction Network
class MaskPredNet(nn.Module):
    def __init__(self, in_channel=768):
        super(MaskPredNet, self).__init__()
        self.ResNet = ResNet()
        self.S1, self.S2, self.S3, self.S4 = self.ResNet.layers()
        self.FPN = FPN(self.S1, self.S2, self.S3, self.S4, 256)

        self.samepadding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(in_channel, 512, kernel_size=3, stride=1)  # This step will
        self.bn1 = nn.BatchNorm2d(512)  # eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.FPN(x)
        out = self.conv1(self.samepadding(x))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(self.samepadding(out))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(self.samepadding(out))
        out = self.bn3(out)
        out = self.relu(out)

        out = self.deconv(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.sigmoid(out)
        # print('mask out shape:', out.shape)
        return out





