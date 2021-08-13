import cv2 as cv
import numpy as np
import os


# gamma transform for image enhancement
def gamma(img, arg):
    img = np.array(img / 255, np.float)
    img = cv.pow(img, arg)
    img = np.array(img * 255, np.uint8)
    return img


if __name__ == '__main__':
    datapath = '../Dataset/iphone/'

    # if img is None:
    #     print('Image Reading Error')

    if datapath[-3:].lower() == 'jpg' or datapath[-3:].lower() == 'png':
        img = cv.imread(datapath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 190, 280)
        _, canny = cv.threshold(canny, 125, 255, cv.THRESH_BINARY_INV)
        cv.imshow('', canny)
        cv.imshow('1', img)
        cv.waitKey(0)

    else:
        print('Batch images test mode')
        img_names = os.listdir(datapath)

        for i, img_name in enumerate(img_names):
            img_path = datapath + '/' + img_name
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            canny = cv.Canny(gray, 190, 280)
            _, canny = cv.threshold(canny, 125, 255, cv.THRESH_BINARY_INV)
            cv.imshow('', canny)
            cv.imshow('1', img)
            cv.waitKey(0)
