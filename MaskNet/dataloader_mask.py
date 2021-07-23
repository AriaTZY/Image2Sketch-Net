from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class MaskDataset(Dataset):
    def __init__(self, path, img_size=224, mask_size=28, aug=True, mode='train'):
        super(MaskDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor()
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize([mask_size, mask_size]),
            transforms.ToTensor()
        ])

        self.mask_size = mask_size
        self.img_sz = img_size
        self.path = path
        self.mode = mode  # train and eval modes require fetching both image and mask, but 'test' mode only need image
        if self.mode == 'test':
            self.img_path = self.path
        else:
            self.img_path = os.path.join(self.path, 'image/')
            self.mask_path = os.path.join(self.path, 'mask/')
        self.img_names = os.listdir(self.img_path)
        self.aug = aug


    def augmentation(self, imgA, imgB, crop_prob=0.5, flip_prob=0.5):
        pic_size = imgB.size[0]
        if np.random.rand() < crop_prob:
            min_window = int(pic_size * 0.7)
            max_window = int(pic_size * 0.99)
            crop_window = np.random.randint(min_window, max_window)
            start_x = np.random.randint(0, pic_size-crop_window-1)
            start_y = np.random.randint(0, pic_size-crop_window-1)
            imgA = imgA.crop((start_x, start_y, start_x+crop_window, start_y+crop_window))
            imgB = imgB.crop((start_x, start_y, start_x+crop_window, start_y+crop_window))

        if np.random.rand() < flip_prob:
            imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)

        return imgA, imgB

    def __getitem__(self, item):
        if self.mode != 'test':
            img = Image.open(self.img_path + self.img_names[item])
            mask = Image.open(self.mask_path + self.img_names[item])

            # Pre-resize is for making input as a square
            img = img.resize((self.img_sz, self.img_sz))
            mask = mask.resize((self.img_sz, self.img_sz))

            # Augmentation. Generally, when training, this is on, vise versa
            if self.aug:
                img, mask = self.augmentation(img, mask)
            mask = mask.convert("L")   # back to 3 channel

            # Post-resize is for making input as a standard size
            imgA = self.transform(img)  # resize and from [0-255] to [0, 1]
            mask_big = self.transform(mask)
            imgB = self.transform_mask(mask)

            return {'A': imgA, 'B': imgB, 'mask': mask_big}

        else:
            img = Image.open(self.img_path + self.img_names[item])
            imgA = self.transform(img)  # resize and from [0-255] to [0, 1]

            return imgA

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
    path = '../MaskDataset/'
    ABdataset = MaskDataset(path, aug=True)

    for i in range(10):
        ret = ABdataset[np.random.randint(0, len(ABdataset))]
        imgA = ret['A'].numpy()
        imgB = ret['B'].numpy()
        mask = ret['mask'].numpy()
        print(mask.shape)
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

