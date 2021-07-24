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



## 3. Usage

### Usage Note:

The network is mainly composed of "Mask Net" and "Sketch Net", among which Mask Net is an auxiliary function, so please keep the background of the input picture as simple as possible, and put the object you want to sketchize in the middle of the image. Otherwise the network will not have any idea of which object should be selected.

#### 1. pipeline.py

This file can run the whole process, i.e., ***Mask Net*** followed by ***Sketch Net***. Output images are in folder "output/Pipeline".

``` 
python pipeline.py --resolution 2 --cuda True --versio  2
```

**--datapath:** The path of folder which includes many or single test image (world images)

**--outpath:** Default value is "output/Pipeline/", you can also assign your own output folder

**--cuda:** Use or not use CUDA

**--version:** Mask Net version, "1" is 28x28 mask size, "2" is 56x56 mask size

**--maskmode:** 'soft' or 'hard', soft means crop foreground object with soft way, 'hard' method will case aliasing, but it will generate more clear boundary when it comes to Sketch Net.

**--softctr:** [0, 1]  float value. 0.3 means the pixel in mask image smaller 0.3 will be set to 0. smaller it is, bigger the object

**--resolution:** 1: 300x300, 2: 424x424, 3: 848x848, 4: 1272x1272.



## Results

High resolution with "soft" mask mode (824x824 input)

![000010](./docs/000010.png)

low resolution with "hard" mask mode, you can see the jagged boundary (424x424 input)

![000010_hard](./docs/000010_hard.png)
