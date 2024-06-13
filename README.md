# MaskTracker -- Hardware accelerated mask detection for IoT applications

This repository contains the results of the MaskTracker project, and is not yet fully functional. Some of the provided files contain bugs and may not run. 

## Contents:
- Pynq-Z2/
  - FINN-generated deployment package
  - Python/
    - Non-threaded Python script 
    - Multithreaded Python script and Jupyter notebook (currently not functional)
  - img/
    - Example pictures

- training/
  - Training script used for cnv-w1a1 (Taken from brevitas_examples/bnn-pynq, contains a few tweaks)
  - models/
    - Pretrained cnv-w1a1 model
    - Pretrained resnet18 model 
  - FINN/
    - end to end FINN compiler notebook (Taken from finn examples, contains a few tweaks)
    - FINN generated performance/resource estimates 

- RPi/
  - UDP Client/server code
  - Raspberry Pi code
  

To recreate the results of this project, a Pynq-Z2 and a Raspberry Pi are required. To recreate the steps of this project, you will need the following resources:

- [FINN (includes Brevitas)](https://github.com/Xilinx/finn)
- [finn_examples](https://github.com/Xilinx/finn-examples)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Masked face dataset](https://github.com/cabani/MaskedFace-Net?tab=readme-ov-file)
- [Artifical face dataset](https://github.com/SelfishGene/SFHQ-dataset)
- [Brevitas BNN_PYNQ (Included with FINN)](https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq)

We additionally used the following resources as inspiration or guidance, but are not necessary to recreate the project:

- [Lightweight face mask detection](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/masked_face)
- [Pytorch reference training scripts](https://github.com/pytorch/vision/blob/main/references/classification/README.md)
- [TripleNet (used as benchmark)](https://github.com/RuiyangJu/TripleNet)
- [PYNQ source code](https://github.com/Xilinx/PYNQ)
