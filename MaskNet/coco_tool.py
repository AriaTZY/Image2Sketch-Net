import numpy as np
from pycocotools.coco import COCO
import cv2 as cv


dataDir = 'F:/DeepLearning/COCO'
dataType = 'val2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

# load a cat image
CatIds = coco.getCatIds(catNms=['bicycle'])
ImgIds = coco.getImgIds(catIds=CatIds)
img_root_path = '{}/{}/'.format(dataDir, dataType)

for i in range(100):
    img_dict = coco.loadImgs(ImgIds[np.random.randint(0, len(ImgIds))])[0]
    img_name = img_dict['file_name']
    img_path = '{}/{}'.format(img_root_path, img_name)
    img = cv.imread(img_path)

    # annotation
    Ann_ids = coco.getAnnIds(imgIds=img_dict['id'], catIds=CatIds)
    ann = coco.loadAnns(Ann_ids)
    seg = ann[0]['segmentation']
    print('part: ', len(seg), 'is crowd:', ann[0]['iscrowd'],
          'area: {}, percentage:{:.2f}%'.format(ann[0]['area'], 100 * ann[0]['area']/(img.shape[0] * img.shape[1])), )
    mask_img = coco.annToMask(ann[0])
    mask_img = np.array(mask_img*255, np.uint8)

    # if len(seg) == 2:
    cv.imshow('', img)
    cv.imshow('mask', mask_img)
    cv.waitKey(0)

    coco.showAnns(ann)




cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

super_cat_nms = [cat['supercategory'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(super_cat_nms)))

# # 统计各类的图片数量和标注框数量
# for cat_name in cat_nms:
#     catId = coco.getCatIds(catNms=cat_name)
#     imgId = coco.getImgIds(catIds=catId)
#     annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
#
#     print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))