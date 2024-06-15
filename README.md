# MaskTracker -- Hardware accelerated mask detection for IoT applications

This repository contains the source code and results of the MaskTracker project, which is still a work in progress. Some of the provided files are still buggy and may not run. 

## Contents
- Pynq-Z2/
  - FINN-generated deployment package for cnv-1w1a
  - Python script 
  - Jupyter notebook (contains non-functional ALS thread)
  - img/
    - Example pictures

- training/
  - models/
    - Pretrained cnv-1w1a model
    - Pretrained resnet18 model 
  - reports/
    - FINN generated performance/resource estimates 

- RPi/
  - UDP Client/server code
  - Raspberry Pi code
  
## Resources

To recreate the results of this project, a Pynq-Z2 and a Raspberry Pi are required, as well as the following (to be installed on the Pynq-Z2): 

- [Lightweight face mask detection](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/masked_face)
- [Caffe SSD + pycaffe](https://github.com/weiliu89/caffe/tree/ssd)
- Wifi dongle or some other means of internet connectivity for the Pynq-Z2

To recreate the steps of this project, you will additionally need the following resources:

- [FINN (includes Brevitas)](https://github.com/Xilinx/finn)
- [finn_examples](https://github.com/Xilinx/finn-examples)
- [Masked face dataset](https://github.com/cabani/MaskedFace-Net?tab=readme-ov-file)
- [Artifical face dataset](https://github.com/SelfishGene/SFHQ-dataset)

We additionally used the following resources as inspiration or guidance, but are not necessary to recreate the project:

- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Pytorch reference training scripts](https://github.com/pytorch/vision/blob/main/references/classification/README.md)
- [TripleNet (used as benchmark)](https://github.com/RuiyangJu/TripleNet)
- [PYNQ source code](https://github.com/Xilinx/PYNQ)

