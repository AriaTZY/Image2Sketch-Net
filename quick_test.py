# 这个文件会把iphone的标准测试图片使用对应的网络快速过一遍，只检测Sketch Net的表现

from network import *
from dataloader_skt import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools import save_network, visualize_result
import argparse
from csv_process import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='PC')
parser.add_argument('--name', type=str, default='', help='model name suffix')
parser.add_argument('--shallow', type=bool, default=False, help='Whether to choose shallow G net')
parser.add_argument('--outpath', type=str, default='output/StandardTest/')
parser.add_argument('--mask', type=bool, default=True)
args = parser.parse_args()

os.makedirs(args.outpath, exist_ok=True)
if args.mode == 'PC':
    cuda = False
else:
    cuda = True

if args.mask:
    data_path = 'input/tmp_test/'
else:
    data_path = 'input/iphone/'

img2skt = True
model_name = args.name

ABdataset = ImageDatasetTest(data_path, img_size=600)
dataloader = DataLoader(dataset=ABdataset, batch_size=1, shuffle=False)

inchannel = 3
outchannel = 1

if args.shallow:  # 浅层网络模型
    print('Using Shallow Net')
    G_net = Generator_Shallow(inchannel, outchannel)
else:
    print('Using deep Net')
    G_net = Generator(inchannel, outchannel)

if cuda:
    G_net = G_net.cuda()

# resume training
if os.path.exists('checkpoints/G_net'+model_name+'.pth'):
    print('loading pre-trained model...')
    G_net.load_state_dict(torch.load('checkpoints/G_net' + model_name + '.pth'))

# or initialization
else:
    print('ERROR, no model found ...')
    raise EOFError

# loss function
L1_loss = torch.nn.L1Loss()  #torch.nn.BCELoss()  #
GAN_loss = torch.nn.BCELoss()

if cuda:
    L1_loss = L1_loss.cuda()
    GAN_loss = GAN_loss.cuda()

iter = 0
loss_G = 0

for i, imgA in enumerate(dataloader):

    if cuda:
        imgA = imgA.cuda()

    gen_B = G_net(imgA)

    print('iter {}'.format(i))

    # =============
    # save output
    # =============
    save_name = args.outpath + str(iter) + '.png'
    sheet = visualize_result(imgA[0], gen_B[0], gen_B[0], width=300, save=save_name)
    # cv.imshow('sheet', sheet)
    # cv.waitKey(0)

    iter += 1



