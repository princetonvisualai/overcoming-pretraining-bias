# [Overwriting Pretrained Bias with Finetuning Data](https://arxiv.org/abs/2303.06167)

## Code
All code can be run through main.py, where the command line flags control what is run
For example, `python3 main.py --bias_type 0 --type 0 1 --experiments 0`

## Packages
- Pandas
- Raytune
- Pytorch
- Torchvision
- Pycocotools

## Datasets used
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [FairFace](https://github.com/joojs/fairface)
- [COCO](https://cocodataset.org/#home)
- [ImageNet](https://www.image-net.org/)
- [Dollar Street](https://mlcommons.org/en/dollar-street/)
- [GeoDE](https://geodiverse-data-collection.cs.princeton.edu/)

## Model Weights used
- [SimCLR](https://github.com/Spijkervet/SimCLR)
- [MoCo](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md)
- [Places](https://github.com/CSAILVision/places365)
- [TorchVision](https://pytorch.org/vision/stable/models/resnet.html)

## Citation

```
@inproceedings{wang2023overwritingpretrainbias,
author = {Angelina Wang and Olga Russakovsky},
title = {Overwriting Pretrained Bias with Finetuning Data},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2023}
}
```
