# horizontally compare different model
from network import *
from dataloader_skt import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools import save_network, visualize_result
import argparse
from csv_process import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='PC')
parser.add_argument('--test_num', type=int, default=5)
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()

inchannel = 3
outchannel = 1
cuda = args.cuda
test_num = args.test_num

if args.mode == 'PC':  data_path = 'C:/Users/tan/Desktop/SketchDataset/image/'
else:  data_path = '../Dataset/iphone/'

# ========================================
#       Model 1: shallow with GAN
# ========================================
model1_last_name = '_shallow_GAN'
model1_full_name = 'G_net' + model1_last_name + '.pth'
G_net_1 = Generator_Shallow(inchannel, outchannel)
G_net_1.load_state_dict(torch.load('checkpoints/' + model1_full_name))
if cuda: G_net_1 = G_net_1.cuda()

# ========================================
#       Model 2: shallow without GAN
# ========================================
model2_last_name = '_shallow_noGAN'
model2_full_name = 'G_net' + model1_last_name + '.pth'
G_net_2 = Generator_Shallow(inchannel, outchannel)
G_net_2.load_state_dict(torch.load('checkpoints/' + model2_full_name))
if cuda: G_net_2 = G_net_2.cuda()

# ========================================
#       Model 3: deep with GAN
# ========================================
model3_last_name = '_deep_GAN'
model3_full_name = 'G_net.pth'
G_net_3 = Generator(inchannel, outchannel)
G_net_3.load_state_dict(torch.load('checkpoints/' + model3_full_name))
if cuda: G_net_3 = G_net_3.cuda()


# load Dataset
Mydataset = ImageDatasetTest(data_path, img_size=424)
dataloader = DataLoader(dataset=Mydataset, batch_size=1, shuffle=True)

# save path
log_path = 'output/model_compare/'
os.makedirs(log_path, exist_ok=True)

# loss function
BCE_loss = torch.nn.BCELoss()
L1_loss = torch.nn.L1Loss()
if cuda:
    L1_loss = L1_loss.cuda()
    BCE_loss = BCE_loss.cuda()

iter = 0


for i, img in enumerate(dataloader):

    if cuda: img = img.cuda()

    skt_1 = G_net_1(img)
    skt_2 = G_net_2(img)
    skt_3 = G_net_3(img)

    img_np = tensor2numpy(img[0])
    skt_1 = tensor2numpy(skt_1[0])
    skt_2 = tensor2numpy(skt_2[0])
    skt_3 = tensor2numpy(skt_3[0])

    cv.imwrite(log_path + str(iter) + '_' + model1_last_name[1:] + '.png', skt_1)
    cv.imwrite(log_path + str(iter) + '_' + model2_last_name[1:] + '.png', skt_2)
    cv.imwrite(log_path + str(iter) + '_' + model3_last_name[1:] + '.png', skt_3)
    cv.imwrite(log_path + str(iter) + '_' + 'image.png', img_np)

    print('No. {} writing done!'.format(iter))

    iter += 1
    if iter >= test_num:
        break



