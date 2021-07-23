# This is for generating the data from COCO DataSet, and CROP the ROI with the biggest mask
import numpy as np
from pycocotools.coco import COCO
import cv2 as cv
import argparse
import os


def image_limit(high, val):
    val = 0 if val < 0 else val
    val = high if val > high else val
    return val


# expand window based on bbox
def expand_zone(img, bbox, expand_range=(1.5, 2), min_width=80, max_ratio=2, always_centre=False):
    img_h, img_w = img.shape[:2]
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    centre_x = bbox_x + bbox_w/2
    centre_y = bbox_y + bbox_h/2

    expand = np.random.randint(expand_range[0]*100, expand_range[1]*100)/100
    # after expand
    if always_centre:  # if "Always centre", means the mask is always at the centre of the image
        half_width = bbox_w / 2
        half_height = bbox_h / 2
        x1 = int(centre_x - half_width * expand)
        y1 = int(centre_y - half_height * expand)
        x2 = int(centre_x + half_width * expand)
        y2 = int(centre_y + half_height * expand)

    else:  # The object(mask) can be the side
        window_w = bbox_w * expand
        window_h = bbox_h * expand
        move_x = bbox_w * (np.random.rand() * (expand - 1) + 0.5)
        move_y = bbox_h * (np.random.rand() * (expand - 1) + 0.5)

        x1 = int(centre_x - move_x)
        y1 = int(centre_y - move_y)
        x2 = int(centre_x + window_w - move_x)
        y2 = int(centre_y + window_h - move_y)

    # Crop if it exceeded the image size
    x1 = image_limit(img_w, x1)
    y1 = image_limit(img_h, y1)
    x2 = image_limit(img_w, x2)
    y2 = image_limit(img_h, y2)

    # Check width-height ratio, avoid some extreme window
    flag = True
    denominator = x2 - x1
    denominator = 1e-5 if denominator <= 0 else denominator
    ratio = (y2-y1)/denominator
    if ratio < 1/max_ratio or ratio > max_ratio:
        flag = False

    # Check minimal window width
    if (x2-x1) < min_width or (y2-y1) < min_width:
        flag = False

    # print('ratio:', (y2-y1)/(x2-x1), ' width:', (y2-y1), ' ', (x2-x1))

    return flag, int(x1), int(y1), int(x2), int(y2)


save_count = 0
val_save_count = 0
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--coco', type=str, default='F:/DeepLearning/COCO/', help='The direction of COCO dataset')
    parse.add_argument('--dataType', type=str, default='val2014', help='Data type')
    parse.add_argument('--save_root', type=str, default='../MaskDataSet/', help='Save root direction')
    parse.add_argument('--val_ratio', type=float, default=0.8, help='The ratio of training data and validation data')
    parse.add_argument('--num', type=int, default=10, help='Generated images for every class, 26 classes are chosen')
    args = parse.parse_args()

    # create output data dictionary
    out_root = args.save_root + '/training/'
    img_save = out_root + '/image/'
    mask_save = out_root + '/mask/'
    fusion_save = out_root + '/fusion/'
    if not os.path.exists(out_root): os.makedirs(out_root)
    if not os.path.exists(img_save): os.makedirs(img_save)
    if not os.path.exists(mask_save): os.makedirs(mask_save)
    if not os.path.exists(fusion_save): os.makedirs(fusion_save)

    # validation
    out_root = args.save_root + '/validation/'
    img_save_val = out_root + '/image/'
    mask_save_val = out_root + '/mask/'
    fusion_save_val = out_root + '/fusion/'
    if not os.path.exists(out_root): os.makedirs(out_root)
    if not os.path.exists(img_save_val): os.makedirs(img_save_val)
    if not os.path.exists(mask_save_val): os.makedirs(mask_save_val)
    if not os.path.exists(fusion_save_val): os.makedirs(fusion_save_val)

    num = args.num
    stop_flag = False

    annFile = '{}/annotations/instances_{}.json'.format(args.coco, args.dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    cats = coco.loadCats(coco.getCatIds())
    wanted_class = [0, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 33, 40, 52, 53,
                    62, 63, 77]

    for k, cat_idx in enumerate(wanted_class):
        cat = cats[cat_idx]
        print('Category: {}/{}, image number: {}'.format(cat_idx, cat['name'], save_count))
        local_count = 0

        # load a cat image
        CatIds = coco.getCatIds(catNms=[cat['name']])
        ImgIds = coco.getImgIds(catIds=CatIds)
        img_root_path = '{}/{}/'.format(args.coco, args.dataType)

        for i in range(len(ImgIds)):
            img_dict = coco.loadImgs(ImgIds[i])[0]
            img_name = img_dict['file_name']
            img_path = '{}/{}'.format(img_root_path, img_name)

            # annotation
            Ann_ids = coco.getAnnIds(imgIds=img_dict['id'], catIds=CatIds)
            anns = coco.loadAnns(Ann_ids)

            # Improve the quality of the training data set. If a picture contains multiple current classes, it means
            # that the quality of each part of the picture is low so discard it.
            tmp_count = 0
            discard_flag = False
            chosen_anns = []
            for ann in anns:
                if ann['category_id'] == CatIds[0]:
                    chosen_anns.append(ann)
                    tmp_count += 1
                if tmp_count >= 3:
                    discard_flag = True
                    break
            # print('There are {} objects in this image'.format(len(anns)), discard_flag)

            # skip this pic if too many objects show on
            if discard_flag:
                continue
            elif len(chosen_anns) == 2:  # if not skip, that must be only one or two objects in this pic
                area1 = chosen_anns[0]['area']
                area2 = chosen_anns[1]['area']
                ratio = max(area1, area2) / min(area1, area2)
                if ratio > 5:  # that means one of them are low-quality, delete that
                    if area1 > area2:  chosen_anns.pop(1)
                    else:  chosen_anns.pop(0)

            for obj_idx in range(len(chosen_anns)):
                ann = anns[obj_idx]

                # Find the bbox
                bbox = ann['bbox']
                x, y, w, h = bbox[:4]
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                mask_img = coco.annToMask(ann)
                mask_img = np.array(mask_img * 255, np.uint8)

                img = cv.imread(img_path)

                # expand bbox and crop the image
                ret, x1, y1, x2, y2 = expand_zone(img, (x, y, w, h), expand_range=(1.5, 2))
                if ret:
                    window = img[y1:y2, x1:x2].copy()
                    mask_window = mask_img[y1:y2, x1:x2].copy()

                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # generate a fusion image
                    mask_window_nor = mask_window.copy()
                    mask_window_nor[mask_window_nor == 0] = 30
                    mask_window_nor = cv.cvtColor(mask_window_nor, cv.COLOR_GRAY2RGB)
                    mask_window_nor = mask_window_nor/255  # normalized
                    fusion_window = np.array(window * mask_window_nor, np.uint8)

                    # save Image and Mask
                    if np.random.rand() > args.val_ratio:  # validation
                        cv.imwrite('{}/{:05d}.png'.format(img_save_val, val_save_count), window)
                        cv.imwrite('{}/{:05d}.png'.format(mask_save_val, val_save_count), mask_window)
                        cv.imwrite('{}/{:05d}.png'.format(fusion_save_val, val_save_count), fusion_window)
                        val_save_count += 1
                    else:
                        cv.imwrite('{}/{:05d}.png'.format(img_save, save_count), window)
                        cv.imwrite('{}/{:05d}.png'.format(mask_save, save_count), mask_window)
                        cv.imwrite('{}/{:05d}.png'.format(fusion_save, save_count), fusion_window)
                        save_count += 1
                    local_count += 1

                    if local_count >= num:
                        stop_flag = True

                    # cv.imshow('origin', img)
                    # cv.imshow('', window)
                    # cv.imshow('mask', mask_window)
                    # cv.waitKey(0)

            if stop_flag:
                stop_flag = False
                # print('Generate {} pairs images'.format(save_count))
                break







