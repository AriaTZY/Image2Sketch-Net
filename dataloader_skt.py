from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, path, img_size=128, aug=True):
        super(ImageDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor()
        ])

        self.img_sz = img_size
        self.path = path
        self.img_path = os.path.join(self.path, 'image/')
        self.skt_path = os.path.join(self.path, 'sketch/')
        self.img_names = os.listdir(self.skt_path)
        self.aug = aug

    def augmentation(self, imgA, imgB, crop_prob=0.5, rotate_prob=0.5, flip_prob=0.5):
        imgA = imgA.resize((imgB.size[0], imgB.size[1]))
        pic_size = imgB.size[0]
        if np.random.rand() < crop_prob:
            min_window = int(pic_size * 0.4)
            max_window = int(pic_size * 0.9)
            crop_window = np.random.randint(min_window, max_window)
            start_x = np.random.randint(0, pic_size-crop_window-1)
            start_y = np.random.randint(0, pic_size-crop_window-1)
            imgA = imgA.crop((start_x, start_y, start_x+crop_window, start_y+crop_window))
            imgB = imgB.crop((start_x, start_y, start_x+crop_window, start_y+crop_window))
            # print('  + augment: crop image, size {}/{}'.format(crop_window, pic_size))

        if np.random.rand() < rotate_prob:
            degree = 30 * (np.random.rand()*2-1)
            imgA = imgA.rotate(degree, fillcolor=(255, 255, 255))
            imgB = imgB.rotate(degree, fillcolor=(255, 255, 255))
            # print('  + augment: rotate image, degree {:.4f}'.format(degree))

        if np.random.rand() < flip_prob:
            imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
            # print('  + augment: flip image')

        return imgA, imgB

    def __getitem__(self, item):
        img = Image.open(self.img_path + self.img_names[item])
        skt = Image.open(self.skt_path + self.img_names[item])

        # Augmentation. Generally, when training, this is on, vise versa
        if self.aug:
            img, skt = self.augmentation(img, skt)
        skt = skt.convert("L")  # image A is sketch, B is picture

        imgA = self.transform(img)  # resize and from [0-255] to [0, 1]
        imgB = self.transform(skt)

        return {'A': imgA, 'B': imgB}

    def __len__(self):
        return len(self.img_names)


# Just for visualization, output format [w, h, 3]
def tensor2numpy(img):
    img = img.detach().cpu().numpy()
    img = img.transpose([1, 2, 0])
    img = np.squeeze(img)
    img = np.array(img * 255, np.uint8)  # [0, 1] -> [0, 255]

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

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

        imgA = cv.cvtColor(imgA, cv.COLOR_RGB2BGR)
        imgB = cv.cvtColor(imgB, cv.COLOR_RGB2BGR)

        sheet[0:h, i*w:(i+1)*w, :] = imgA
        sheet[h:, i*w:(i+1)*w, :] = imgB

    # If Don't Save, Show it
    if save is None:
        cv.imshow('f', sheet)
        cv.waitKey(0)
    else:
        cv.imwrite(save, sheet)


if __name__ == '__main__':
    path = 'C:/Users/tan/Desktop/SketchDataset/'
    ABdataset = ImageDataset(path)
    ret = ABdataset[23]
    imgA = ret['A'].numpy()
    imgB = ret['B'].numpy()
    imgA = imgA.transpose([1, 2, 0])
    imgB = imgB.transpose([1, 2, 0])
    imgB = np.squeeze(imgB)

    print(imgB.shape)
    plt.subplot(121)
    plt.imshow(imgA)
    plt.subplot(122)
    plt.imshow(imgB, 'gray')
    plt.show()


    # dataloader = DataLoader(dataset=ABdataset, batch_size=3, shuffle=True)
    #
    # for i, batch in enumerate(dataloader):
    #     data = batch['A']
    #     print(data.shape)
    #     w, h = data.shape[-2:]
    #     print(w, h)
    #
    #     visualize_batch_input(batch, 10, save='try.png')
    #     break

