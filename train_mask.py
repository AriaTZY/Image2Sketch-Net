import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tools import visualize_result, visualize_batch_result
import time
import os
from MaskNet.dataloader_mask import MaskDataset
import tensorboardX


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='MaskDataSet/', help='The path of mask dataset')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--inter', type=int, default=200, help='Save model and images interval')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--exp', type=int, default=0, help='bigger than 0 means using tensorboard')
    parser.add_argument('--resume_iter', type=int, default=0, help='bigger than 0 means using tensorboard')
    args = parser.parse_args()

    cuda = args.cuda
    interval = args.inter
    model_version = args.version
    print('Now Running Model version: #', model_version)

    if args.exp >= 0:
        print('USE TensorboardX')
        logger_path = 'tensorboard/{}/'.format(args.exp)
        if not os.path.exists(logger_path): os.makedirs(logger_path)
        logger = tensorboardX.SummaryWriter(logger_path)

    if model_version == 1:
        from MaskNet.MaskNetwork import ResNet, MaskPredNet
        mask_size = 28
    elif model_version == 2:
        from MaskNet.MaskNetwork2 import MaskPredNet
        mask_size = 56

    # ==== DataLoader =======
    data_path = args.datapath + 'training/'
    dataset_training = MaskDataset(data_path, mask_size=mask_size, aug=True)
    dataloader_training = DataLoader(dataset=dataset_training, batch_size=args.batch_size, shuffle=True)

    data_path = args.datapath + 'validation/'
    dataset_val = MaskDataset(data_path, mask_size=mask_size, aug=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True)

    # ====== Net ===========
    if model_version == 1:
        resnet = ResNet()
        MaskNet = MaskPredNet(1024, resnet)
        model_path = 'checkpoints/MaskNet.pth'
    elif model_version == 2:
        MaskNet = MaskPredNet(in_channel=768)
        model_path = 'checkpoints/MaskNet_v2.pth'
    if cuda: MaskNet = MaskNet.cuda()

    # resume training
    if os.path.exists(model_path):
        print('resume training, loading pre-trained model...')
        MaskNet.load_state_dict(torch.load(model_path))

    # ======= log =======
    log_path = 'logs/'
    if model_version == 1:
        img_log_path = log_path + 'image_model_v1/'
    else:
        img_log_path = log_path + 'image_model_v2/'

    if not os.path.exists(img_log_path):
        os.makedirs(img_log_path)

    optimizer = torch.optim.Adam(MaskNet.parameters(), lr=1e-5)
    print('\n Info: There are {} image pairs in the Dataset'.format(len(dataloader_training)))
    time.sleep(1)

    # ======== Training =========
    global_count = args.resume_iter
    start_time = time.time()
    for epoch in range(args.epoch):
        for iter, batch_data in enumerate(dataloader_training):
            img = batch_data['A']
            mask = batch_data['B']  # the mask with the size of 28x28
            mask_show = batch_data['mask']  # the mask with the original size (bigger)

            if cuda:
                img = img.cuda()
                mask = mask.cuda()

            pred_mask = MaskNet(img)
            loss = F.binary_cross_entropy(pred_mask, mask)
            loss.backward()
            optimizer.step()

            global_count += 1

            print('epoch {} iter {}, loss {}, time {:.4f}s'.format(epoch, iter, loss, time.time()-start_time))

            # write into the tensorboardX
            if args.exp >= 0 and global_count % 50 == 0:
                logger.add_scalar('training loss', loss.item(), global_step=global_count)

            # save images
            if global_count % interval == 0:
                save_name = '{}/epoch_{}_iter_{}_round_{}.png'.format(img_log_path, epoch, iter, global_count)
                visualize_batch_result(img, pred_mask, mask, width=224, save=save_name)
                torch.save(MaskNet.cpu().state_dict(), model_path)
                if cuda:
                    MaskNet.cuda()

            # model evaluation
            if (global_count - 1) % 100 == 0:
                for _, batch_data in enumerate(dataloader_val):
                    img = batch_data['A']
                    mask = batch_data['B']  # the mask with the small size

                    if cuda:
                        img = img.cuda()
                        mask = mask.cuda()

                    MaskNet.eval()
                    pred_mask = MaskNet(img)
                    loss = F.binary_cross_entropy(pred_mask, mask)
                    loss = loss.detach().cpu().numpy()
                    print('validation loss:', loss)
                    MaskNet.train()

                    if args.exp >= 0:
                        logger.add_scalar('val loss', loss.item(), global_step=global_count)

                    break

