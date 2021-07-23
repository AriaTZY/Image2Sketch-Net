import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tools import visualize_result, visualize_batch_result, post_process_mask
import time
import os
from MaskNet.dataloader_mask import MaskDataset
import numpy as np
import cv2 as cv


# Evaluation step require the ground true mask image to calculate some quantitative indices
def eval(net, args, mask_sz):
    interval = 10
    cuda = args.cuda
    root_path = args.datapath
    img_path = root_path + 'image/'
    mask_path = root_path + 'mask/'
    assert os.path.exists(img_path), 'Require Image folder'
    assert os.path.exists(mask_path), 'Require Mask folder'

    # ==== DataLoader =======
    data_path = args.datapath
    dataset = MaskDataset(data_path, aug=True, mode='val', mask_size=mask_sz)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # ======== eval =========
    loss_list = []
    for iter, batch_data in enumerate(dataloader):
        img = batch_data['A']
        mask = batch_data['B']  # the mask with the size of 28x28
        mask_show = batch_data['mask']  # the mask with the original size (bigger)

        if cuda:
            img = img.cuda()
            mask = mask.cuda()

        pred_mask = net(img)
        loss = F.binary_cross_entropy(pred_mask, mask)
        loss_list += [loss]

        print('image {}/{}, loss {}'.format(iter, len(dataloader), loss))

        if iter % interval == 0:
            # save_name = '{}/epoch_{}_iter_{}_round_{}.png'.format(img_log_path, epoch, iter, global_count)
            sheet = visualize_result(img[0], pred_mask[0], mask[0], width=300, save=None)
            cv.imshow('result', sheet)
            cv.waitKey(0)

    loss_numpy = np.array(loss_list)
    mean_loss = loss_numpy.mean()
    print('Evaluation done, there are {} image pairs, mean loss is {:.6f}'.format(len(dataloader), mean_loss))


# Evaluation step require the ground true mask image to calculate some quantitative indices
def test(net, args, num=5, mask_sz=28):
    cuda = args.cuda
    img_path = args.datapath

    # ==== DataLoader =======
    dataset = MaskDataset(img_path, aug=False, mode='test', mask_size=mask_sz)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # ======== test =========
    for iter, img in enumerate(dataloader):

        if cuda:
            img = img.cuda()

        pred_mask = net(img)

        print('image {}/{}'.format(iter, len(dataloader)))

        root = 'output/MaskNet/epoch600_v2/'
        if not os.path.exists(root): os.makedirs(root)
        save_name = '{}/{:04d}.png'.format(root, iter)

        post_process_mask(img[0], pred_mask[0], save=save_name)

        if iter >= num:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='you can choose from "test" or "eval" mode')
    parser.add_argument('--datapath', type=str, default='MaskDataSet/', help='The path of Test data, it should')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--version', type=int, default=2)
    args = parser.parse_args()

    cuda = args.cuda
    model_version = args.version
    print('Now Running Model version: #', model_version)

    if model_version == 1:
        from MaskNet.MaskNetwork import ResNet, MaskPredNet
        mask_size = 28
    elif model_version == 2:
        from MaskNet.MaskNetwork2 import MaskPredNet
        mask_size = 56

    args.datapath = 'E:/Postgraduate/Dataset/WithBack/image/'

    # ======================
    # ====== Network =======
    # ======================
    if model_version == 1:
        resnet = ResNet()
        MaskNet = MaskPredNet(1024, resnet)
        # model_path = 'checkpoints/MaskNet.pth'
        model_path = 'checkpoints/MaskNet_500epoch.pth'

    else:
        MaskNet = MaskPredNet(in_channel=768)
        model_path = 'checkpoints/MaskNet_v2_stage2.pth'

    if cuda: MaskNet = MaskNet.cuda()

    # resume training
    if os.path.exists(model_path):
        print('loading pre-trained model...')
        MaskNet.load_state_dict(torch.load(model_path))

    MaskNet.eval()

    if args.mode == 'test':
        test(MaskNet, args, mask_size)
    elif args.mode == 'eval':
        eval(MaskNet, args, mask_size)
    else:
        print('please check your MODE parameter!')
