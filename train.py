from network import *
from dataloader_skt import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools import save_network, visualize_result
import argparse
from csv_process import *


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='PC')
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--name', type=str, default='_deep_GAN_L1', help='model name suffix')
parser.add_argument('--shallow', type=bool, default=False, help='Whether to choose shallow G net')
parser.add_argument('--load', type=str, default=True, help='Whether to choose shallow G net')
parser.add_argument('--log_int', type=int, default=300, help='interval of write log file')
args = parser.parse_args()

load_model = True if args.load == 'True' else False

epochs = 350000
if args.mode == 'PC':
    data_path = 'C:/Users/tan/Desktop/SketchDataset/'
else:
    data_path = '../Dataset/'

batch_size = 5
lr = 0.0001  # 0.01
img2skt = True
cuda = args.cuda
model_name = args.name

log_path = 'logs/sketch_oneshot/'
os.makedirs(log_path, exist_ok=True)

ABdataset = ImageDataset(data_path, img_size=424, image_folder='image')
dataloader = DataLoader(dataset=ABdataset, batch_size=batch_size, shuffle=True)

if img2skt:
    inchannel = 3
    outchannel = 1
else:
    inchannel = 1
    outchannel = 3

if args.shallow:  # 浅层网络模型
    print('Using Shallow Net')
    G_net = Generator_Shallow(inchannel, outchannel)
else:
    print('Using deep Net')
    G_net = Generator(inchannel, outchannel)
D_net = Discriminator()

if cuda:
    G_net = G_net.cuda()
    D_net = D_net.cuda()

# resume training
if os.path.exists('checkpoints/G_net'+model_name+'.pth') and load_model:
    print('resume training, loading pre-trained model...')
    G_net.load_state_dict(torch.load('checkpoints/G_net' + model_name + '.pth'))
    D_net.load_state_dict(torch.load('checkpoints/D_net' + model_name + '.pth'))

# or initialization
else:
    # 由于随机数在服务器上坏了，我并不准备使用直接初始化的方法，而是读入在PC上提前生成好的初始化权重
    print('Initializing model ...')
    # if args.shallow:
    #     G_net.load_state_dict(torch.load('checkpoints/G_net_init_shallow.pth'))
    # else:
    #     G_net.load_state_dict(torch.load('checkpoints/G_net_init_deep.pth'))
    # D_net.load_state_dict(torch.load('checkpoints/D_net_init.pth'))

    # def weight_init(m):
    #     classname = m.__class__.__name__  # 得到网络层的名字，如ConvTranspose2d
    #     if classname.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
    #         m.weight.data.normal_(0.0, 1.5)
    #         # m.weight.data.uniform_(0, 1.7)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.normal_(1.0, 0.2)
    #         # m.weight.data.uniform_(0, 1)
    #         m.bias.data.fill_(0)
    # print('initialize net weights')
    # G_net.apply(weight_init)
    # D_net.apply(weight_init)

# print(D_net)
G_optimizer = torch.optim.Adam(G_net.parameters(), lr=lr)
D_optimizer = torch.optim.Adam(D_net.parameters(), lr=lr)

# loss function
L1_loss = torch.nn.L1Loss()  # torch.nn.BCELoss() # torch.nn.L1Loss()
GAN_loss = torch.nn.BCELoss()

if cuda:
    L1_loss = L1_loss.cuda()
    GAN_loss = GAN_loss.cuda()

# log writer
log = CSVManager(model_name[1:], head=['BCE_loss', 'G_GAN', 'D_GAN'])

iter = 0

for epoch in range(epochs):
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

        valid = Variable(torch.Tensor(np.ones([gen_B.shape[0], 1, 1, 1])), requires_grad=False)
        false = Variable(torch.Tensor(np.zeros([gen_B.shape[0], 1, 1, 1])), requires_grad=False)
        if cuda:
            valid = valid.cuda()
            false = false.cuda()

        # =============
        # train G net
        # =============
        G_optimizer.zero_grad()

        genAB = torch.cat([imgA, gen_B], dim=1)  # cat along 'channel' axis
        pred_fake = D_net(genAB)

        loss_G_GAN = Patch_GAN_loss(pred_fake, True, GAN_loss, cuda)
        # loss_G_GAN = GAN_loss(pred_fake, valid)
        loss_G_L1 = L1_loss(gen_B, imgB)
        loss_G = loss_G_GAN + loss_G_L1

        loss_G.backward()
        G_optimizer.step()

        # =============
        # train D net
        # =============
        D_optimizer.zero_grad()

        realAB = torch.cat([imgA, imgB], dim=1)
        pred_real_D = D_net(realAB)
        loss_D_real = Patch_GAN_loss(pred_real_D, True, GAN_loss, cuda)
        # loss_D_real = GAN_loss(pred_real_D, valid)

        fakeAB = torch.cat([imgA, gen_B.detach()], dim=1)  # it is important, detach preventing training G net when training D net
        pred_fake_D = D_net(fakeAB)
        loss_D_fake = Patch_GAN_loss(pred_fake_D, False, GAN_loss, cuda)
        # loss_D_fake = GAN_loss(pred_fake_D, false)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        D_optimizer.step()

        iter += 1

        # print('batch {}, iter {}, G_loss:{:.4f}, D_loss:{:.8f}'.format(epoch, i, loss_G, loss_D))
        print('batch {}, iter {}, G_loss:{:.4f}, L1_loss:{:.8f}, GAN_loss:{:.8f}, D_loss:{:.8f}'.format(epoch, i, loss_G, loss_G_L1, loss_G_GAN, loss_D))

        # =============
        # save model
        # =============
        if iter % 150 == 0:
            print('>>>> save models')
            save_network(G_net, 'G_net' + model_name + '.pth', cuda)
            save_network(D_net, 'D_net' + model_name + '.pth', cuda)
        if iter % args.log_int == 0 or iter == 1:
            log.write_data(epoch, iter, [loss_G_L1.item(), loss_G_GAN.item(), loss_D.item()])
            log.generate_graph([0, 1, 2])

        # =============
        # save output
        # =============
        if iter % 100 == 0:
            print('>>>> save tmp pics')
            sheet = visualize_result(imgA[0], gen_B[0], imgB[0], width=300, save='result.png')
        if iter % 2000 == 0 or iter == 1:
            print('>>>> save in log folder')
            for i in range(min(batch_size, 2)):
                save_name = log_path + '/iter_{}_sample_{}'.format(iter, i) + '.png'
                sheet = visualize_result(imgA[i], gen_B[i], imgB[i], width=300, save=save_name)



