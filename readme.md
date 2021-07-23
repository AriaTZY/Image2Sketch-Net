# Image to Sketch Network

## 1. File Explanation

##### Sketch Net

* **dataloader_skt:** The data-loader for Sketch Net, including data augmentation function
* **network:** The architecture of Sketch Net, resemble to pix2pix (cGAN) network
* **test:** Solo test for Sketch Net
* **tools:** Some functions to visualize/save images/process mask image

##### Mask Net

* **MaskNet/dataloader_mask:** Dataloader for mask net
* **MaskNet/DataGeneration:** Generate training dataset from COCO, it followed by class
* **MaskNet/MaskNetwork:** Shallow Mask Net, output mask size is 28 x 28
* **MaskNet/MaskNetwork2** Deepper Mask Net, output mask size is 56 x 56
* **train_mask, test_mask:** similar to above



## 2. Checkpoints Note:

* **MasNet_v2_notcentre: **The network trained on not centered mask image data. trained on 24/07/2021
* **MaskNet_v2_stage1**: The model trained on centered mask data and saved on halfway, seems not overfitting
* **MaskNet_v2_stage2:** The model saved after the full training, seems to have a serious overfitting



## 3. MaskDataSet

