from network import *
from dataloader_skt import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools import save_network, visualize_result
import argparse
from csv_process import *

# 这段是用代码生成mask后的自然图片并保存
# pathimg = '../Dataset/iphone/'
# pathmask = '../Dataset/mask/'
# pathout = '../Dataset/out/'
# os.makedirs(pathout, exist_ok=True)
# name_list = os.listdir(pathimg)
# for i in name_list:
#     img = cv.imread(pathimg + i)
#     mask = cv.imread(pathmask + i[:-4] + '.png', 1)
#     out = cv.bitwise_and(img, mask)
#     out = cv.bitwise_or(cv.bitwise_not(mask), out)
#     cv.imwrite(pathout + i[:-4] + '.png', out)
#     # cv.imshow('', out)
#     # cv.waitKey(0)
#
# print('finsi')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='PC')
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--name', type=str, default='', help='model name suffix')
parser.add_argument('--shallow', type=bool, default=False, help='Whether to choose shallow G net')
parser.add_argument('--idx', type=int, default=0, help='from which index to start output')
parser.add_argument('--outpath', type=str, default='output/shallow/')
args = parser.parse_args()

os.makedirs(args.outpath, exist_ok=True)
if args.mode == 'PC':
    data_path = 'C:/Users/tan/Desktop/SketchDataset/'
else:
    data_path = '../Dataset/'

img2skt = True
cuda = args.cuda
model_name = args.name

ABdataset = ImageDataset(data_path, img_size=424, aug=False, image_folder='image')
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
    print('resume training, loading pre-trained model...')
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

for i, batch in enumerate(dataloader):
    if i < args.idx:
        continue

    imgA = batch['A']  # image
    imgB = batch['B']  # sketch

    if cuda:
        imgA = imgA.cuda()
        imgB = imgB.cuda()

    gen_B = G_net(imgA)

    # =============
    # G net
    # =============
    loss_G_L1 = L1_loss(gen_B, imgB)  #

    # white_weight = 0.3
    # white_weight *= 2
    # weight = torch.tensor([2-white_weight, white_weight])  # [black weight, white weight]
    # weight_ = weight[imgB.data.view(-1).long()].view_as(imgB)
    # if cuda: weight_ = weight_.cuda()
    # loss_G_L1 = loss_G_L1 * weight_
    # loss_G_L1 = loss_G_L1.mean()

    loss_G += loss_G_L1.item()  # + 1 * (1 - torch.mean(torch.abs(gen_B)))  # 这里增添了正则项

    print('iter {}, mean_loss:{:.4f}, L1_loss:{:.8f}'.format(i, loss_G/(iter+0.00001), loss_G_L1))

    # =============
    # save output
    # =============

    print('>>>> save tmp pics')
    save_name = args.outpath + str(iter) + '.png'
    sheet = visualize_result(imgA[0], gen_B[0], imgB[0], width=300, save=save_name)
    # cv.imshow('sheet', sheet)
    # cv.waitKey(0)

    iter += 1



