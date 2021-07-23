import torch
import torch.nn as nn
import numpy as np


class conv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, step=2):
        super(conv, self).__init__()
        pad = int((kernel-1)/2)
        layer = []
        layer += [nn.Conv2d(in_feat, out_feat, kernel, stride=step, padding=pad)]
        layer += [nn.BatchNorm2d(out_feat, 0.8)]
        layer += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*layer)

    def forward(self, x):
        x = self.model(x)
        return x


class deconv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, step=2, pad=1):
        super(deconv, self).__init__()
        layer = []
        layer += [nn.ConvTranspose2d(in_feat, out_feat, kernel, stride=step, padding=pad)]
        layer += [nn.BatchNorm2d(out_feat, 0.8)]
        layer += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*layer)

    def forward(self, x):
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self, inchannel=3, outchannel=1):
        super(Generator, self).__init__()

        # Encoder
        self.down1 = conv(inchannel, 48, 5, 2)  # 424 -> 212
        self.flat1_1 = conv(48, 128, 3, 1)
        self.flat1_2 = conv(128, 128, 3, 1)
        self.flat1_3 = conv(128, 128, 3, 1)

        self.down2 = conv(128, 256, 3, 2)  # 212 -> 106
        self.flat2_1 = conv(256, 256, 3, 1)
        self.flat2_2 = conv(256, 256, 3, 1)

        # bottle neck
        self.down3 = conv(256, 256, 3, 2)  # 106 -> 53
        self.flat3_1 = conv(256, 256, 3, 1)
        self.flat3_2 = conv(256, 512, 3, 1)
        self.flat3_3 = conv(512, 512, 3, 1)
        self.flat3_4 = conv(512, 256, 3, 1)

        # Decoder
        self.up1 = deconv(256, 256, 4, 2, 1)  # 53 -> 106
        self.flat4_1 = conv(256, 256, 3, 1)
        self.flat4_2 = conv(256, 128, 3, 1)

        self.up2 = deconv(128, 128, 4, 2, 1)  # 106 -> 212
        self.flat5_1 = conv(128, 128, 3, 1)
        self.flat5_2 = conv(128, 48, 3, 1)

        self.up3 = deconv(48, 48, 4, 2, 1)  # 212 -> 424
        self.flat6_1 = conv(48, 24, 3, 1)
        self.flat6_2 = conv(24, outchannel, 3, 1)

        self.final = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(self.down1, self.flat1_1, self.flat1_2, self.flat1_3,
                                   self.down2, self.flat2_1, self.flat2_2,
                                   self.down3, self.flat3_1, self.flat3_2, self.flat3_3, self.flat3_4,
                                   self.up1, self.flat4_1, self.flat4_2,
                                   self.up2, self.flat5_1, self.flat5_2,
                                   self.up3, self.flat6_1, self.flat6_2, self.final)

    def forward(self, x):
        output = self.model(x)
        # print('generate:', output.shape, output.max(), output.min())
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        multi = 2  # Enlarge dims when shirking the image size
        in_dims = 48

        # default input patch size is [(3+1) x 424 x 424]
        self.down1 = conv(4, in_dims, 5, 2)    # -> 212 (dim:48)
        self.flat1_1 = conv(in_dims, in_dims, 3, 1)    # -> 212 (48)

        out_dim = in_dims * multi
        self.down2 = conv(in_dims, out_dim, 5, 2)   # -> 106 (96)
        self.flat2_1 = conv(out_dim, out_dim, 3, 1)  # -> 106 (96)

        in_dims = out_dim
        out_dim = in_dims * multi
        self.down3 = conv(in_dims, out_dim, 5, 2)  # -> 53 (192)
        self.flat3_1 = conv(out_dim, out_dim, 3, 1)  # -> 53 (192)
        self.flat3_2 = conv(out_dim, out_dim, 3, 1)  # -> 53 (192)

        in_dims = out_dim
        out_dim = in_dims * multi
        self.down4 = conv(in_dims, out_dim, 3, 4)  # -> 14 (384)
        self.flat4_1 = conv(out_dim, out_dim, 3, 1)  # -> 14 (384)
        self.flat4_2 = conv(out_dim, out_dim, 3, 1)  # -> 14 (384)
        self.flat4_3 = conv(out_dim, out_dim, 3, 1)  # -> 14 (384)

        in_dims = out_dim
        out_dim = 512
        self.down5 = conv(in_dims, out_dim, 3, 2)  # -> 7 (512)
        self.flat5_1 = conv(out_dim, out_dim, 1, 1)  # -> 7 (512)
        self.down6 = conv(out_dim, 1, 3, 2)  # -> 4 (1)

        self.convolution = nn.Sequential(self.down1, self.flat1_1,
                                         self.down2, self.flat2_1,
                                         self.down3, self.flat3_1, self.flat3_2,
                                         self.down4, self.flat4_1, self.flat4_2, self.flat4_3,
                                         self.down5, self.flat5_1, self.down6)
        self.downsample = nn.AvgPool2d(4, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convolution(x)
        res = self.downsample(x)
        res = self.sigmoid(res)
        # print('res shape:', res.max(), res.min())
        return res





