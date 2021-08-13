import cv2 as cv
import matplotlib.pyplot as plt
from dataloader_skt import tensor2numpy
import numpy as np
import torch


# put input, output, ground true in one row to compare
def visualize_result(input, output, GT, width=None, save=None):

    input = tensor2numpy(input)
    output = tensor2numpy(output)
    GT = tensor2numpy(GT)

    if len(input.shape) == 2: input = cv.cvtColor(input, cv.COLOR_GRAY2RGB)
    if len(output.shape) == 2: output = cv.cvtColor(output, cv.COLOR_GRAY2RGB)
    if len(GT.shape) == 2: GT = cv.cvtColor(GT, cv.COLOR_GRAY2RGB)

    if width is None:
        print('width is None')
        width = input.shape[0]

    if not input.shape == output.shape == GT.shape or width is not None:
        'The shapes are not equal'
        input = cv.resize(input, (width, width))
        output = cv.resize(output, (width, width))
        GT = cv.resize(GT, (width, width))

    sheet = np.zeros([width, width*3, 3], np.uint8)
    sheet[:, :width] = input
    sheet[:, width:width*2] = output
    sheet[:, width*2:] = GT

    # If Don't Save, Show it
    if save is None:
        # cv.imshow('result', sheet)
        # cv.waitKey(0)
        # plt.imshow(sheet)
        # plt.show()
        pass
    else:
        sheet = cv.cvtColor(sheet, cv.COLOR_RGB2BGR)
        cv.imwrite(save, sheet)

        # plt.imshow(sheet)
        # plt.show()

    return sheet


def visualize_batch_result(input_batch, output_batch, GT_batch, width: int=200, save=None):
    batch_num = input_batch.shape[0]

    # 6 is the max limit
    if batch_num > 6: batch_num = 6

    row_num = int(np.ceil(batch_num / 2))
    sheet = np.zeros([width * row_num, width * 3 * 2, 3], np.uint8)

    for row in range(row_num):
        for i in range(2):
            idx = row * 2 + i
            if idx >= batch_num:
                break
            sheet_part = visualize_result(input_batch[idx], output_batch[idx], GT_batch[idx], width, save=None)
            sheet[row*width:(row+1)*width, 3*width*i:3*width*(i+1)] = sheet_part

    # If Don't Save, Show it
    if save is None:
        plt.imshow(sheet)
        plt.show()
    else:
        sheet = cv.cvtColor(sheet, cv.COLOR_RGB2BGR)
        cv.imwrite(save, sheet)

    return sheet


# Post Process the predicted mask, input should be a Torch.tensor
def post_process_mask(img, mask, threshold=0.4, save=None):
    # ====================
    # Tensor to Numpy
    # ====================
    mask = mask.detach().cpu().numpy()
    mask = mask.transpose([1, 2, 0])
    mask = np.squeeze(mask)
    mask = np.array(mask * 255, np.uint8)  # [0, 1] -> [0, 255]
    if len(mask.shape) == 3:
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

    img = img.detach().cpu().numpy()
    img = img.transpose([1, 2, 0])
    img = np.array(img * 255, np.uint8)  # [0, 1] -> [0, 255]

    # ====================
    # Binary
    # ====================
    ret, mask = cv.threshold(mask, int(255 * threshold), 255, cv.THRESH_BINARY)
    mask = cv.resize(mask, (img.shape[1], img.shape[0]))

    # ====================
    # Generate a fusion image
    # ====================
    mask_fusion = mask.copy()
    mask_fusion[mask_fusion == 0] = 5
    mask_fusion = cv.cvtColor(mask_fusion, cv.COLOR_GRAY2RGB)
    mask_fusion = mask_fusion / 255  # normalized
    fusion = np.array(img * mask_fusion, np.uint8)

    # ====================
    # Create sheet
    # ====================
    width = img.shape[1]
    sheet = np.zeros([width, width * 3, 3], np.uint8)
    sheet[:, :width] = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    sheet[:, width:width * 2] = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    sheet[:, width * 2:] = cv.cvtColor(fusion, cv.COLOR_RGB2BGR)

    if save is None:
        cv.imshow('result', sheet)
        cv.waitKey(0)

    else:
        cv.imwrite(save, sheet)

    return sheet


def save_network(net, name, cuda):
    path = 'checkpoints/' + name
    torch.save(net.cpu().state_dict(), path)
    if cuda:
        net.cuda()



