# PIPELINE: input an world image and output an sketch of the object in the image
from network import *
from dataloader_skt import *
from MaskNet.dataloader_mask import MaskDataset
import matplotlib.pyplot as plt
import argparse


# for single image test, no need to call "Database" class, use this
def image2tensor(img, img_sz=224, mode='cv'):
    if mode == 'PIL':
        transform = transforms.Compose([
            transforms.Resize([img_sz, img_sz]),
            transforms.ToTensor()
        ])
        img = transform(img)  # resize and from [0-255] to [0, 1]
    elif mode == 'cv':
        img = cv.resize(img, (img_sz, img_sz))
        img = img/255
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)

    img = torch.unsqueeze(img, dim=0)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='MaskDataSet/', help='The path of Test data, it can be a '
                                                                             'folder or an image file name')
    parser.add_argument('--outpath', type=str, default='output/Pipeline/')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--maskmode', type=str, default='soft')  # or binary
    args = parser.parse_args()

    "For non-cmd run"
    # args.datapath = 'E:/Postgraduate/Dataset/WithBack/image/'
    args.datapath = 'E:/Postgraduate/Dataset/iphone/'

    cuda = args.cuda
    model_version = args.version
    print('Now Running Mask Model version: #', model_version)

    # ======================
    #  Network Preparation
    # ======================
    " Mask Net "
    if model_version == 1:
        from MaskNet.MaskNetwork import ResNet, MaskPredNet
        mask_size = 28
        resnet = ResNet()
        MaskNet = MaskPredNet(1024, resnet)
        # model_path = 'checkpoints/MaskNet.pth'
        model_path = 'checkpoints/MaskNet_500epoch.pth'

    else:
        from MaskNet.MaskNetwork2 import MaskPredNet
        mask_size = 56
        MaskNet = MaskPredNet(in_channel=768)
        model_path = 'checkpoints/MaskNet_v2_stage1.pth'

    " Sketch Net "
    G_net = Generator(inchannel=3, outchannel=1)

    if cuda:
        MaskNet = MaskNet.cuda()
        G_net = G_net.cuda()

    # Load Model
    if os.path.exists(model_path):
        print('loading pre-trained model...')
        MaskNet.load_state_dict(torch.load(model_path))
        G_net.load_state_dict(torch.load('checkpoints/G_net.pth'))

    MaskNet.eval()
    G_net.eval()

    data_path = args.datapath

    def single_pipeline(img_path):
        # img = Image.open(img_path)
        # w, h = img.size()
        # img_tensor = image2tensor(img, mode='PIL')

        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img_tensor = image2tensor(img)

        if cuda: img_tensor = img_tensor.cuda()

        # ============================
        #          Mask Net
        # ============================
        pred_mask = MaskNet(img_tensor)

        # ============================
        #     Crop Image by Mask
        # ============================
        pred_mask = torch.squeeze(pred_mask)
        mask_np = pred_mask.detach().cpu().numpy()
        white_background = np.zeros([h, w, 3], np.uint8)
        white_background = white_background + 255

        # " soft method "
        if args.maskmode == 'soft':
            mask_np[mask_np > 0.5] = 1
            mask_np = cv.resize(mask_np, dsize=(w, h))
            mask_np_3c = mask_np[:, :, np.newaxis]
            cropped_img = np.array(img * mask_np_3c, np.uint8)
            cropped_img = cropped_img + np.array(white_background * (1 - mask_np_3c), np.uint8)

        # " binary method "
        else:
            mask_np[mask_np > 0.5] = 1
            mask_np[mask_np <= 0.5] = 0
            mask_np = np.array(mask_np * 255, np.uint8)
            mask_np = cv.resize(mask_np, dsize=(w, h))
            ret, mask_np_inv = cv.threshold(mask_np, 120, 255, cv.THRESH_BINARY_INV)
            cropped_img = cv.bitwise_and(img, img, mask=mask_np)
            cropped_img = cropped_img + cv.bitwise_and(white_background, white_background, mask=mask_np_inv)

        # cv.imshow('mask', mask_np)
        # cv.imshow('img', cropped_img)
        # cv.waitKey(0)

        # ============================
        #         Sketch Net
        # ============================
        cropped_img_tensor = image2tensor(cropped_img, img_sz=424, mode='cv')
        if cuda: cropped_img_tensor = cropped_img_tensor.cuda()
        pred_sketch = G_net(cropped_img_tensor)
        pred_sketch = torch.squeeze(pred_sketch)
        pred_sketch = pred_sketch.detach().cpu().numpy()
        pred_sketch = np.array(pred_sketch * 255, np.uint8)
        pred_sketch = cv.resize(pred_sketch, (w, h))

        # ============================
        #       Save or Display
        # ============================
        sheet = np.zeros([h, w * 3, 3], np.uint8)
        sheet[:, :w] = img
        sheet[:, w:2 * w] = cropped_img
        sheet[:, 2 * w:] = cv.cvtColor(pred_sketch, cv.COLOR_GRAY2RGB)

        sheet = cv.cvtColor(sheet, cv.COLOR_RGB2BGR)

        return sheet


    # ==================== single image test =====================
    if data_path[-3:] == 'jpg' or data_path[-3:] == 'png':
        print('Single image test mode')
        sheet = single_pipeline(data_path)
        '''save image'''
        root = args.outpath
        if not os.path.exists(root): os.makedirs(root)
        img_name = data_path[data_path.rfind('/') + 1:data_path.rfind('.')]
        save_name = '{}/{}.png'.format(root, img_name)
        cv.imwrite(save_name, sheet)

    # ==================== batch image test ==================
    else:
        print('Batch images test mode')
        out_root = args.outpath
        if not os.path.exists(out_root): os.makedirs(out_root)
        img_names = os.listdir(data_path)

        for i, img_name in enumerate(img_names):
            img_path = data_path + '/' + img_name
            sheet = single_pipeline(img_path)
            ''' save images '''
            img_name = img_name[:img_name.rfind('.')]
            save_name = '{}/{}.png'.format(out_root, img_name)
            cv.imwrite(save_name, sheet)
            print('Processing {}/{}, save in {}'.format(i, len(img_names), save_name))






