from network import *
from dataloader_skt import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools import save_network, visualize_result
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='PC')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--num', type=int, default=5)
parser.add_argument('--shuffle', type=bool, default=True)
args = parser.parse_args()

if args.mode == 'PC':
    data_path = 'C:/Users/tan/Desktop/SketchDataset/'
    data_path = 'E:/Postgraduate/Dataset/Test/'
    data_path = 'E:/Postgraduate/Dataset/WithBack/'
else:
    data_path = '../Dataset/'
out_root = 'output/'
if os.path.exists(out_root): os.makedirs(out_root, exist_ok=True)

batch_size = 1
img2skt = True
cuda = args.cuda

ABdataset = ImageDataset(data_path, img_size=424, aug=False)
dataloader = DataLoader(dataset=ABdataset, batch_size=batch_size, shuffle=args.shuffle)

if img2skt:
    inchannel = 3
    outchannel = 1
else:
    inchannel = 1
    outchannel = 3

G_net = Generator(inchannel, outchannel)
D_net = Discriminator()

if cuda:
    G_net = G_net.cuda()
    D_net = D_net.cuda()

G_net.eval()
D_net.eval()

# resume training
if os.path.exists('checkpoints/G_net.pth'):
    print('resume training, loading pre-trained model...')
    G_net.load_state_dict(torch.load('checkpoints/G_net.pth'))
    D_net.load_state_dict(torch.load('checkpoints/D_net.pth'))
else:
    raise ValueError('Cannot find pre-trained model!')


for i, batch in enumerate(dataloader):
    if img2skt:
        imgA = batch['A']  # image
        imgB = batch['B']  # sketch
    else:
        imgA = batch['B']  # sketch
        imgB = batch['A']  # image

    if cuda:
        imgA = imgA.cuda()
        imgB = imgB.cuda()

    gen_B = G_net(imgA)

    # =============
    # save output
    # =============
    print('>>>> save pics ', i)
    sheet = visualize_result(imgA[0], gen_B[0], imgB[0], '{}/result_{}.png'.format(out_root, i))

    if i + 1 >= args.num:
        break


