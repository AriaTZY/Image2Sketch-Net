import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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

        self.model = nn.Sequential(self.down1, self.flat1_1, self.flat1_2, self.flat1_3,  # 0-3
                                   self.down2, self.flat2_1, self.flat2_2,  # 4-6
                                   self.down3, self.flat3_1, self.flat3_2, self.flat3_3, self.flat3_4,  # 7-11
                                   self.up1, self.flat4_1, self.flat4_2,  # 12-14
                                   self.up2, self.flat5_1, self.flat5_2,  # 15-17
                                   self.up3, self.flat6_1, self.flat6_2, self.final)  # 18-21

    def forward(self, x):
        output = self.model(x)

        # =========================
        #  Average hidden layer
        # =========================
        # show = x
        # layer_choice = 21  # change here to visualize the layer you want
        # for i, layer in enumerate(self.model):
        #     show = layer(show)
        #     if i == layer_choice:
        #         break
        # show = show.detach().cpu().numpy()
        # mean_feature_map = []
        # for i in range(show.shape[1]):
        #     tmp = show[0, i, :, :]
        #     tmp = tmp[np.newaxis, :, :]
        #     mean_feature_map.append(tmp)
        # mean_feature_map = np.concatenate(mean_feature_map, axis=0)
        # mean_feature_map = np.mean(mean_feature_map, axis=0)
        # plt.imshow(mean_feature_map, 'gray')
        # plt.axis('off')
        # plt.show()
        # print(mean_feature_map.shape)
        # exit()

        # =========================
        #  visualize hidden layer
        # =========================
        # show = x
        # layer_choice = 17  # change here to visualize the layer you want
        # for i, layer in enumerate(self.model):
        #     show = layer(show)
        #     if i == layer_choice:
        #         break
        # show = show.detach().cpu().numpy()
        #
        # # import os, cv2
        # # export_path = 'C:/Users/tan/Desktop/visualization/layer_{}/'.format(layer_choice)
        # # os.makedirs(export_path, exist_ok=True)
        #
        # for j in range(5):  # loop 5 times
        #     for i in range(min(10, show.shape[1])):
        #         plt.subplot(2, 5, i+1)
        #         choice = np.random.randint(0, show.shape[1])
        #         img = show[0, choice, :, :]
        #         # cv2.imwrite(export_path + str(i) + '.png', np.array(255*(img-np.min(img))/(np.max(img)-np.min(img)), np.uint8))
        #         plt.imshow(img)
        #         name = 'layer {}, channel {}/{}'.format(layer_choice, choice, show.shape[1])
        #         plt.title(name)
        # plt.show()

        return output


class Generator_Shallow(nn.Module):
    def __init__(self, inchannel=3, outchannel=1):
        super(Generator_Shallow, self).__init__()

        # Encoder
        self.down1 = conv(inchannel, 48, 5, 2)  # 424 -> 212
        self.flat1_1 = conv(48, 128, 3, 1)
        self.flat1_2 = conv(128, 128, 3, 1)
        self.flat1_3 = conv(128, 128, 3, 1)

        self.up3 = deconv(128, 48, 4, 2, 1)  # 212 -> 424
        self.flat6_1 = conv(48, 24, 3, 1)
        self.flat6_2 = conv(24, outchannel, 3, 1)

        self.final = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(self.down1, self.flat1_1, self.flat1_2, self.flat1_3,
                                   self.up3, self.flat6_1, self.flat6_2, self.final)

    def forward(self, x):
        output = self.model(x)

        # =========================
        #  visualize hidden layer
        # =========================
        # show = x
        # layer_choice = 20  # change here to visualize the layer you want
        # for i, layer in enumerate(self.model):
        #     show = layer(show)
        #     if i == layer_choice:
        #         break
        # show = show.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # for j in range(5):  # loop 5 times
        #     for i in range(min(10, show.shape[1])):
        #         plt.subplot(2, 5, i+1)
        #         choice = np.random.randint(0, show.shape[1])
        #         plt.imshow(show[0, choice, :, :])
        #         name = 'layer {}, channel {}/{}'.format(layer_choice, i+1, show.shape[1])
        #         plt.title(name)
        # plt.show()

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)  # 212 -> 106

        # default input patch size is [(3+1) x 424 x 424]
        self.down1 = conv(4, 64, 3, 1)    # -> 212 (dim:64)
        self.down1half = conv(64, 64, 3, 1)    # -> 212 (dim:64)

        # MAX POOLING 212 -> 106
        self.down2 = conv(64, 128, 3, 1)   # -> 106 (128)

        # MAX POOLING  # 106 -> 53
        self.down3 = conv(128, 256, 3, 1)  # -> 53 (256)
        self.flat3_1 = conv(256, 256, 3, 1)  # -> 53 (256)

        # MAX POOLING  # 53 -> 26
        self.down4 = conv(256, 512, 3, 1)  # -> 26 (512)
        self.flat4_1 = conv(512, 512, 3, 1)  # -> 14 (512)
        self.flat4_2 = conv(512, 512, 3, 1)  # -> 14 (384)

        # MAX POOLING  # 26 -> 13
        self.down5 = conv(512, 512, 3, 1)  # -> 13 (512)
        self.flat5_1 = conv(512, 512, 3, 1)  # -> 13 (512)
        self.flat5_2 = conv(512, 512, 3, 1)  # -> 13 (512)

        # MAX POOLING  # 13 -> 6
        self.flat6 = conv(512, 512, 3, 1)  # 6->6

        " Alternative last-layer structure, this is the version mentioned in my dissertation "
        # self.flat6 = conv(512, 128, 3, 2)  # 6->3
        # self.flat7 = conv(512, 1, 3, 1)  # 3->1

        # 可以算作VGG12
        self.convolution = nn.Sequential(self.down1, self.maxpool, self.down1half, self.maxpool,
                                         self.down2, self.maxpool,
                                         self.down3, self.flat3_1, self.maxpool,
                                         self.down4, self.flat4_1, self.flat4_2, self.maxpool,
                                         self.down5, self.flat5_1, self.flat5_2, self.maxpool,
                                         self.flat6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convolution(x)
        res = self.sigmoid(x)
        print('res.shape', res.shape)
        # print('res shape:', res.shape)
        return res


# # # First version
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         multi = 2  # Enlarge dims when shirking the image size
#         in_dims = 48
#
#         # default input patch size is [(3+1) x 424 x 424]
#         self.down1 = conv(4, in_dims, 5, 2)    # -> 212 (dim:48)
#         self.flat1_1 = conv(in_dims, in_dims, 3, 1)    # -> 212 (48)
#
#         out_dim = in_dims * multi
#         self.down2 = conv(in_dims, out_dim, 5, 2)   # -> 106 (96)
#         self.flat2_1 = conv(out_dim, out_dim, 3, 1)  # -> 106 (96)
#
#         in_dims = out_dim
#         out_dim = in_dims * multi
#         self.down3 = conv(in_dims, out_dim, 5, 2)  # -> 53 (192)
#         self.flat3_1 = conv(out_dim, out_dim, 3, 1)  # -> 53 (192)
#         self.flat3_2 = conv(out_dim, out_dim, 3, 1)  # -> 53 (192)
#
#         in_dims = out_dim
#         out_dim = in_dims * multi
#         self.down4 = conv(in_dims, out_dim, 3, 4)  # -> 14 (384)
#         self.flat4_1 = conv(out_dim, out_dim, 3, 1)  # -> 14 (384)
#         self.flat4_2 = conv(out_dim, out_dim, 3, 1)  # -> 14 (384)
#         self.flat4_3 = conv(out_dim, out_dim, 3, 1)  # -> 14 (384)
#
#         in_dims = out_dim
#         out_dim = 512
#         self.down5 = conv(in_dims, out_dim, 3, 2)  # -> 7 (512)
#         self.flat5_1 = conv(out_dim, out_dim, 1, 1)  # -> 7 (512)
#         self.down6 = conv(out_dim, 1, 3, 2)  # -> 4 (1)
#
#         self.convolution = nn.Sequential(self.down1, self.flat1_1,
#                                          self.down2, self.flat2_1,
#                                          self.down3, self.flat3_1, self.flat3_2,
#                                          self.down4, self.flat4_1, self.flat4_2, self.flat4_3,
#                                          self.down5, self.flat5_1, self.down6)
#         self.downsample = nn.AvgPool2d(4, 1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.convolution(x)
#         res = self.downsample(x)
#         res = self.sigmoid(res)
#         # print('res shape:', res.shape)
#         return res


from torch.autograd import Variable
def Patch_GAN_loss(tensor, flag, loss, cuda):
    if flag:
        # single_label = torch.tensor(1.0)
        single_label = Variable(torch.Tensor([1.0]), requires_grad=False)
    else:
        # single_label = torch.tensor(0.0)
        single_label = Variable(torch.Tensor([0.0]), requires_grad=False)
    if cuda: single_label = single_label.cuda()

    return loss(tensor, single_label.expand_as(tensor))


def weighted_L1(gt, genB, weight=0.3):
    white_loss = torch.mean(gt * torch.abs(genB - gt))
    black_loss = torch.mean((1-gt) * torch.abs(genB - gt))
    return white_loss * weight + black_loss * (2-weight)


class weighted_BCE(nn.Module):
    def __init__(self, white_weight=0.5):
        super(weighted_BCE, self).__init__()
        self.weight = nn.Parameter(torch.Tensor([white_weight]), requires_grad=False)

    def forward(self, pred, gt):
        weight = self.weight.expand_as(pred)
        loss = - (weight*gt*torch.log2(pred) + (1-weight)*(1-gt)*(torch.log2(1 - pred)))
        return Variable(2 * torch.mean(loss))


# white_w = 0.5: means white part share the same importance of black line
# def weighted_BCE(gt, pred, white_w=0.49, cuda=False):
#     weight = Variable(torch.Tensor([white_w]), requires_grad=False)
#     weight = weight.expand_as(pred)
#     if cuda:
#         weight = weight.cuda()
#     loss = -(weight.mul_(gt).mul_(torch.log(pred)).add_( (1-weight).mul_(1-gt).mul_(torch.log(1-pred)) ))
#     return 2*torch.mean(loss)
