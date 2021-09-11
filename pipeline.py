# PIPELINE: input an world image and output an sketch of the object in the image
from network import *
from dataloader_skt import *
from MaskNet.dataloader_mask import MaskDataset
import matplotlib.pyplot as plt
import argparse
from virtual_sketching.test_vectorization import *


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


# gamma transform for image enhancement
def gamma(img, arg):
    img = np.array(img / 255, np.float)
    img = cv.pow(img, arg)
    img = np.array(img * 255, np.uint8)
    return img


# inverse image in HSV space
def inverse_img(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = 255-hsv[:, :, 2]

    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='input/iphone/', help='The path of Test data, it can be a '
                                                                             'folder or an image file name')
    parser.add_argument('--outpath', type=str, default='output/Test888/')
    parser.add_argument('--cuda', type=str, default='False')
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--maskmode', type=str, default='soft')  # or binary
    parser.add_argument('--softctr', type=float, default=0.5, help='soft control')  # smaller value can expose more object part
    parser.add_argument('--resolution', type=int, default=1)  # 1, 2, 3, 4 x 424 levels or 0 means 800 high resolution
    parser.add_argument('--background', type=str, default='True', help='whether to run mask net to crop foreground')
    parser.add_argument('--vec_input_sz', type=int, default=600, help='resize the sketch image to this size and input to vectorize net')
    parser.add_argument('--vec_switch', type=str, default='False', help='whether to do the vectorize')
    parser.add_argument('--merge', type=str, default='False', help='whether to use positive+negative and merge strategy')
    parser.add_argument('--gamma', type=str, default='True', help='whether to use gamma enhance')
    args = parser.parse_args()

    "For non-cmd run"
    # args.datapath = 'F:/IMAGE_DATABASE/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_003137.jpg'
    # args.datapath = '../Dataset/iphone/IMG_0036.JPG'

    # translate bool value
    args_merge = True if args.merge == 'True' else False
    args_gamma = True if args.gamma == 'True' else False
    args_background = True if args.background == 'True' else False
    args_vec_switch = True if args.vec_switch == 'True' else False

    cuda = True if args.cuda == 'True' else False
    model_version = args.version
    print('Now Running Mask Model version: #', model_version)

    # ======================
    #    Create folders
    # ======================
    gallery_folder = args.outpath + 'gallery/'
    os.makedirs(gallery_folder, exist_ok=True)
    sketch_folder = args.outpath + 'sketch/'
    os.makedirs(sketch_folder, exist_ok=True)
    if args_vec_switch:
        vector_folder = args.outpath + 'vector/'
        gif_folder = args.outpath + 'gif/'
        os.makedirs(vector_folder, exist_ok=True)
        os.makedirs(gif_folder, exist_ok=True)

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
    G_net_pth = 'checkpoints/G_net_x600.pth' if args.resolution == 0 else 'checkpoints/G_net.pth'

    if cuda:
        MaskNet = MaskNet.cuda()
        G_net = G_net.cuda()

    # Load Model
    if os.path.exists(model_path):
        print('loading pre-trained model...')
        MaskNet.load_state_dict(torch.load(model_path))
        G_net.load_state_dict(torch.load(G_net_pth))

    MaskNet.eval()
    G_net.eval()

    data_path = args.datapath

    def core_process(back, img, resized_size):
        h, w = img.shape[:2]
        # img_tensor = image2tensor(img, resized_size)
        # if cuda: img_tensor = img_tensor.cuda()

        # If run Mask Net
        if back:
            # ============================
            #          Mask Net
            # ============================
            img_tensor_for_mask = image2tensor(img)  # mask net's input require smaller image
            if cuda: img_tensor_for_mask = img_tensor_for_mask.cuda()
            pred_mask = MaskNet(img_tensor_for_mask)

            # ============================
            #     Crop Image by Mask
            # ============================
            pred_mask = torch.squeeze(pred_mask)
            mask_np = pred_mask.detach().cpu().numpy()
            white_background = np.zeros([h, w, 3], np.uint8)
            white_background = white_background + 255

            # " soft method "
            if args.maskmode == 'soft':
                mask_np[mask_np > args.softctr] = 1
                mask_np = cv.resize(mask_np, dsize=(w, h))
                mask_np_3c = mask_np[:, :, np.newaxis]
                cropped_img = np.array(img * mask_np_3c, np.uint8)
                cropped_img = cropped_img + np.array(white_background * (1 - mask_np_3c), np.uint8)

            # " binary method "
            else:
                mask_np[mask_np > args.softctr] = 1
                mask_np[mask_np <= args.softctr] = 0
                mask_np = np.array(mask_np * 255, np.uint8)
                mask_np = cv.resize(mask_np, dsize=(w, h))
                ret, mask_np_inv = cv.threshold(mask_np, 120, 255, cv.THRESH_BINARY_INV)
                cropped_img = cv.bitwise_and(img, img, mask=mask_np)
                cropped_img = cropped_img + cv.bitwise_and(white_background, white_background, mask=mask_np_inv)

            # when do mask net, no need to do gamma enhancement, but sketch needed!
            if args_gamma: cropped_img = gamma(cropped_img, 2.0)
            cropped_img_tensor = image2tensor(cropped_img, img_sz=resized_size, mode='cv')

        # No Mask Net mode
        else:
            cropped_img = img
            if args_gamma: cropped_img = gamma(cropped_img, 2.0)
            cropped_img_tensor = image2tensor(cropped_img, resized_size)

        # ============================
        #         Sketch Net
        # ============================
        if cuda: cropped_img_tensor = cropped_img_tensor.cuda()
        pred_sketch = G_net(cropped_img_tensor)
        pred_sketch = torch.squeeze(pred_sketch)
        pred_sketch = pred_sketch.detach().cpu().numpy()
        pred_sketch = np.array(pred_sketch * 255, np.uint8)
        pred_sketch = cv.resize(pred_sketch, (w, h))

        return cropped_img, pred_sketch


    # pipeline, including reading, processing, display
    def single_pipeline(back, img_path, merge=args_merge):
        # img size after resize
        if args.resolution == 0: resized_size = 600
        else: resized_size = args.resolution * 424

        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # ============================
        #       Core process
        # ============================
        h, w = img.shape[:2]
        cropped_img, pred_sketch = core_process(back, img, resized_size)
        if merge:  # merge mode usually is used in animal object
            inv_img = inverse_img(img)  # 255 - img
            _, pred_sketch2 = core_process(back, inv_img, resized_size)
            pred_sketch = cv.bitwise_and(pred_sketch, pred_sketch2)
            # save inter-media step result
            # name_start = img_path[::-1].find('/')
            # name_end = img_path[::-1].find('.')
            # img_name = img_path[-name_start:-(name_end + 1)]
            # cv.imwrite(args.outpath + img_name + '_invImg.png', inv_img)
            # cv.imwrite(args.outpath + img_name + '_invSkt.png', pred_sketch2)
            # cv.imwrite(args.outpath + img_name + '_posSkt.png', pred_sketch)

        # ============================
        #        Vectorize
        # ============================
        # save the result after sketchized, no matter do vectorization or not
        saved_pred_sketch = cv.resize(pred_sketch, (args.vec_input_sz, int(args.vec_input_sz / pred_sketch.shape[1] * pred_sketch.shape[0])))
        name_start = img_path[::-1].find('/')
        name_end = img_path[::-1].find('.')
        img_name = img_path[-name_start:-(name_end + 1)]
        saved_pred_sketch = gamma(saved_pred_sketch, 3)  # gamma can make gray to black
        cv.imwrite(sketch_folder + img_name + '.png', saved_pred_sketch)

        if args_vec_switch:
            sketch2vector(sketch_folder, args.outpath, img_name + '.png', 1, model_base_dir='checkpoints/snapshot/')

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
    if data_path[-3:].lower() == 'jpg' or data_path[-3:].lower() == 'png':
        print('Single image test mode')
        sheet = single_pipeline(args_background, data_path)
        '''save image'''
        root = gallery_folder
        if not os.path.exists(root): os.makedirs(root)
        img_name = data_path[data_path.rfind('/') + 1:data_path.rfind('.')]
        save_name = '{}/{}.png'.format(root, img_name)
        cv.imwrite(save_name, sheet)
        print('Save successfully, path:', save_name)

    # ==================== batch image test ==================
    else:
        print('Batch images test mode')
        out_root = gallery_folder
        if not os.path.exists(out_root): os.makedirs(out_root)
        img_names = os.listdir(data_path)

        for i, img_name in enumerate(img_names):
            img_path = data_path + '/' + img_name
            sheet = single_pipeline(args_background, img_path)
            ''' save images '''
            img_name = img_name[:img_name.rfind('.')]
            save_name = '{}/{}.png'.format(out_root, img_name)
            cv.imwrite(save_name, sheet)
            print('Processing {}/{}, save in {}'.format(i, len(img_names), save_name))

            # live display mode
            # resized_w = int(400 / sheet.shape[0] * sheet.shape[1])
            # sheet = cv.resize(sheet, (resized_w, 400))
            # cv.imshow('', sheet)
            # cv.waitKey(0)



