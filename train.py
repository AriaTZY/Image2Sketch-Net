from network import *
from dataloader_skt import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tools import save_network, visualize_result
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='PC')
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()

epochs = 10000
# data_path = 'E:\Postgraduate\PyTorch-GAN-master\data\edges2shoes/train/'
if args.mode == 'PC':
    data_path = 'C:/Users/tan/Desktop/SketchDataset/'
else:
    data_path = '../Dataset/'
batch_size = 5
lr = 0.0001
img2skt = True
# cuda = True if torch.cuda.is_available() else False
cuda = args.cuda

ABdataset = ImageDataset(data_path, img_size=424)
dataloader = DataLoader(dataset=ABdataset, batch_size=batch_size, shuffle=True)

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

# resume training
if os.path.exists('checkpoints/G_net.pth'):
    print('resume training, loading pre-trained model...')
    G_net.load_state_dict(torch.load('checkpoints/G_net.pth'))
    # D_net.load_state_dict(torch.load('checkpoints/D_net.pth'))

# print(D_net)
G_optimizer = torch.optim.Adam(G_net.parameters(), lr=lr)
D_optimizer = torch.optim.Adam(D_net.parameters(), lr=lr*10)

# loss function
L1_loss = torch.nn.L1Loss()
GAN_loss = torch.nn.BCELoss()

if cuda:
    L1_loss = L1_loss.cuda()
    GAN_loss = GAN_loss.cuda()

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

        loss_G_GAN = GAN_loss(pred_fake, valid)
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
        loss_D_real = GAN_loss(pred_real_D, valid)

        fakeAB = torch.cat([imgA, gen_B.detach()], dim=1)
        pred_fake_D = D_net(fakeAB)
        loss_D_fake = GAN_loss(pred_fake_D, false)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        D_optimizer.step()

        iter += 1

        # print('batch {}, iter {}, G_loss:{:.4f}, D_loss:{:.8f}, {}, {}'.format(epoch, i, loss_G, loss_D, loss_D_real, loss_D_fake))
        print('batch {}, iter {}, G_loss:{:.4f}, D_loss:{:.8f}'.format(epoch, i, loss_G, loss_D))

        # =============
        # save model
        # =============
        if iter % 150 == 0:
            print('>>>> save models')
            save_network(G_net, 'G_net.pth', cuda)
            save_network(D_net, 'D_net.pth', cuda)

        # =============
        # save output
        # =============
        if iter % 50 == 0:
            print('>>>> save pics')
            visualize_batch_input(batch, 5, 'try.png')
            sheet = visualize_result(imgA[0], gen_B[0], imgB[0], width=300, save='result.png')


