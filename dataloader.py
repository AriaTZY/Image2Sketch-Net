from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import cv2 as cv
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, path, img_size=128):
        super(ImageDataset, self).__init__()
        self.transformRGB = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.transformGray = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(0.5, (0.3))
        ])

        self.path = path
        self.img_names = os.listdir(self.path)

    def __getitem__(self, item):
        img = Image.open(self.path + self.img_names[item])
        w, h = img.size
        imgA = img.crop((0, 0, w/2, h))
        imgB = img.crop((w/2, 0, w, h))
        imgA = imgA.convert("L")  # image A is sketch, B is picture

        imgA = self.transformGray(imgA)
        imgB = self.transformRGB(imgB)

        return {'A': imgA, 'B': imgB}

    def __len__(self):
        return len(self.img_names)


def tensor2numpy(img, mode='RGB'):
    img = img.detach().cpu().numpy()
    img = img.transpose([1, 2, 0])

    if mode == 'RGB':
        img = img * np.array([[[0.229, 0.224, 0.225]]])  # inverse std
        img = img + np.array([[[0.485, 0.456, 0.406]]])  # inverse mu
    elif mode == 'L':
        img = img * np.array([[[0.3]]])  # inverse std
        img = img + np.array([[[0.5]]])  # inverse mu
        img = np.squeeze(img)
    img = np.array(img * 255, np.uint8)  # [0, 1] -> [0, 255]
    return img


# Visualize the data with the format [B, C, W, H],
# But this function is only valid for input batch due to its special normalization parameters
def visualize_batch_input(batch_tensor, show_num, save=None):
    batch_num = batch_tensor['A'].shape[0]
    batch_A = batch_tensor['A']
    batch_B = batch_tensor['B']
    w, h = batch_A.shape[-2:]

    num = show_num if show_num < batch_num else batch_num
    sheet = np.zeros([h * 2, w * num, 3], np.uint8)

    for i in range(num):
        imgA = batch_A[i, :, :, :]
        imgB = batch_B[i, :, :, :]
        imgA = tensor2numpy(imgA)
        imgB = tensor2numpy(imgB)

        sheet[0:h, i*w:(i+1)*w, :] = imgA
        sheet[h:, i*w:(i+1)*w, :] = imgB

    # If Don't Save, Show it
    if save is None:
        cv.imshow('f', sheet)
        cv.waitKey(0)
    else:
        cv.imwrite(save, sheet)


# Visualize the data with the format [B, C, W, H]
# This function is valid for the output from sigmoid function
def visualize_batch_output(batch_tensor, show_num, save=None):
    batch_num = batch_tensor.shape[0]
    w, h = batch_tensor.shape[-2:]

    num = show_num if show_num < batch_num else batch_num
    sheet = np.zeros([h * 2, w * num, 3], np.uint8)

    def tensor2numpy(img):
        img = img.detach().cpu().numpy()
        img = img.transpose([1, 2, 0])

        img = img * np.array([[[0.229, 0.224, 0.225]]])  # inverse std
        img = img + np.array([[[0.485, 0.456, 0.406]]])  # inverse mu
        img = np.array(img * 255, np.uint8)  # [0, 1] -> [0, 255]
        return img

    for i in range(num):
        imgA = batch_A[i, :, :, :]
        imgB = batch_B[i, :, :, :]
        imgA = tensor2numpy(imgA)
        imgB = tensor2numpy(imgB)

        sheet[0:h, i*w:(i+1)*w, :] = imgA
        sheet[h:, i*w:(i+1)*w, :] = imgB

    # If Don't Save, Show it
    if save is None:
        cv.imshow('f', sheet)
        cv.waitKey(0)
    else:
        cv.imwrite(save, sheet)


# put input, output, ground true in one row to compare
def visualize_result(input, output, GT, mode='RLL', save=None):
    assert len(mode) == 3, 'The mode should be like "RLL", "LRR"'
    dict = {'R': 'RGB', 'L': 'L'}
    mode1 = dict[mode[0]]
    mode2 = dict[mode[1]]
    mode3 = dict[mode[2]]

    input = tensor2numpy(input, mode1)
    output = tensor2numpy(output, mode2)
    GT = tensor2numpy(GT, mode3)

    if len(input.shape) == 2: input = cv.cvtColor(input, cv.COLOR_GRAY2RGB)
    if len(output.shape) == 2: output = cv.cvtColor(output, cv.COLOR_GRAY2RGB)
    if len(GT.shape) == 2: GT = cv.cvtColor(GT, cv.COLOR_GRAY2RGB)

    assert input.shape == output.shape == GT.shape, 'The shapes are not equal'

    width = input.shape[0]

    sheet = np.zeros([width, width*3, 3], np.uint8)
    sheet[:, :width] = input
    sheet[:, width:width*2] = output
    sheet[:, width*2:] = GT

    # If Don't Save, Show it
    if save is None:
        # cv.imshow('result', sheet)
        # cv.waitKey(0)
        plt.imshow(sheet)
        plt.show()
    else:
        cv.imwrite(save, sheet)

    return sheet


if __name__ == '__main__':
    img = cv.imread('result.png')
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

    path = 'E:\Postgraduate\PyTorch-GAN-master\data\edges2shoes/train/'
    ABdataset = ImageDataset(path)
    dataloader = DataLoader(dataset=ABdataset, batch_size=3, shuffle=True)

    for i, batch in enumerate(dataloader):
        data = batch['A']
        print(data.shape)
        w, h = data.shape[-2:]
        print(w, h)

        visualize_batch_input(batch, 10, save='try.png')
        break



    # img_names = os.listdir(path)
    # img = Image.open(path + img_names[0])
    # img.save('try.jpg')
    # import cv2 as cv
    # img_np = cv.imread('try.jpg')
    # img = Image.fromarray(img_np, 'RGB')
    # img.show()

