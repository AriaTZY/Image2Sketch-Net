# This file can process videos into sketch videos
import argparse
import os
import cv2 as cv
from network import *
from pipeline import image2tensor, gamma
import numpy as np


"""
python video_sketch.py --unit s -b 0 -e 30 --cuda True --vpath input/videos/F1.mp4 --resolution 2
"""

parser = argparse.ArgumentParser()
parser.add_argument('--vpath', type=str, default='input/videos/Anne-Marie.mp4', help='Videos path, end with ".mp4" or other format')
parser.add_argument('--outpath', type=str, default='output/videos/')
parser.add_argument('--unit', type=str, default='s', help='can be "s" or "f", indicating second or frame as unit')
parser.add_argument('--start', '-b', type=int, default=5, help='start frame or second')
parser.add_argument('--end', '-e', type=int, default=10, help='end frame or second')
parser.add_argument('--cuda', type=str, default='False')
parser.add_argument('--resolution', type=int, default=1)  # 1, 2, 3, 4 x 424 levels or 0 means 800 high resolution
parser.add_argument('--background', type=str, default='False', help='whether to run mask net to crop foreground')
parser.add_argument('--gamma', type=str, default='True', help='whether to use gamma enhance')
args = parser.parse_args()

# ======================
#    Parse Parameters
# ======================
args_gamma = True if args.gamma == 'True' else False
cuda = True if args.cuda == 'True' else False
background = True if args.background == 'True' else False
max_width = 800


# Some simple works to crop image to a square or adjust size
def image_process(frame):
    h, w = frame.shape[:2]
    if h > max_width:  # Too big
        ratio = h / max_width
        frame = cv.resize(frame, dsize=ratio)

    if w / h > 1.2:  # Aspect Ratio too large, resize will destroy image content, so crop
        h, w = frame.shape[:2]
        centre = int(w / 2)
        frame = frame[:, centre - (h // 2):centre + (h // 2), :]
    return frame


def core_process(G_net, MaskNet, back, img, resized_size):
    h, w = img.shape[:2]

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

        # " soft thresholding "
        mask_np[mask_np > 0.5] = 1
        mask_np = cv.resize(mask_np, dsize=(w, h))
        mask_np_3c = mask_np[:, :, np.newaxis]
        cropped_img = np.array(img * mask_np_3c, np.uint8)
        cropped_img = cropped_img + np.array(white_background * (1 - mask_np_3c), np.uint8)

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


def save_video(frame_folder, random_name, count, out_path='output.avi', win_size=(480, 480), fps=20):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(out_path + 'output.avi', fourcc, fps=fps, frameSize=win_size)

    for i in range(count):
        load_frame_name = '{}/{}_{}_{}.png'.format(frame_folder, random_name, i, 'skt')
        try:
            frame = cv.imread(load_frame_name)
            frame = cv.resize(frame, win_size)
            out.write(frame)
        except:
            break

    out.release()


if __name__ == '__main__':
    # ======================
    #    Create folders
    # ======================
    out_folder = args.outpath
    frame_tmp_folder = out_folder + '/tmp_frame/'  # 用于存储处理后的每一帧
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(frame_tmp_folder, exist_ok=True)
    random_name = np.random.randint(0, 999999)

    # ======================
    #  Network Preparation
    # ======================
    " Mask Net "
    from MaskNet.MaskNetwork2 import MaskPredNet

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

    # ======================
    #    Video Processing
    # ======================
    cap = cv.VideoCapture(args.vpath)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_cnt = 0
    save_cnt = 0
    resized_size = 600 if args.resolution == 0 else args.resolution * 424

    while cap.isOpened():
        # ============================
        #      Start/End Control
        # ============================
        factor = fps if args.unit == 's' else 1
        if frame_cnt/factor < args.start:
            cap.read()
            frame_cnt += 1
            continue
        elif frame_cnt/factor > args.end:
            break

        print('Processing frame {}/{}'.format(save_cnt, (args.end-args.start)*factor))
        ret, frame = cap.read()
        frame = image_process(frame)

        # ============================
        #    Framework Processing
        # ============================
        print('Network Processing ...')
        ratio = frame.shape[0]/frame.shape[1]  # height/width
        cropped_img, pred_sketch = core_process(G_net, MaskNet, background, frame, resized_size)
        frame = pred_sketch

        'Save frame, release RAM'
        save_frame_name = '{}/{}_{}_{}.png'.format(frame_tmp_folder, random_name, save_cnt, 'skt')
        cv.imwrite(save_frame_name, frame)
        save_cnt += 1
        frame_cnt += 1

        # cv.imshow('frame', frame)
        # # cv.imshow('video', cropped_img)
        # if cv.waitKey(25) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv.destroyAllWindows()

    'Output Videos'
    save_video(frame_tmp_folder, random_name, frame_cnt, out_path=args.outpath,
               win_size=(resized_size, resized_size), fps=fps)
    print('Save Done!')



