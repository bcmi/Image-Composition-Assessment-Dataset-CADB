

> # Image Composition Assessment Dataset
> Welcomes to the offical homepage of Image Composition Assessment DataBase (**CADB**) !
> Image composition assessment aims to assess the overall composition quality of a given image, which is crucial in aesthetic assessment.
> To support the research on this task, we contribute the first image composition assessment dataset. Furthermore, we we propose a composition assessment network **SAMP-Net** 
> with a novel Saliency-Augmented Multi-pattern Pooling (**SAMP**) module, which can perform more favorably than previous aesthetic assessment approaches.
> This work has been accepted by BMVC 2021 ([paper](https://arxiv.org/pdf/2104.03133.pdf)).  

**Table of Contents**

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Dataset](#dataset)
  - [Introduction](#introduction)
  - [Download](#download)
- [Method Overview](#method-overview)
  - [Motivation](#motivation)
  - [SAMP-Net](#samp-net)
- [Results](#results)
- [Code Usage](#code-usage)
  - [Training](#training)
  - [Testing](#testing)
- [Citation](#citation)

# Dataset

## Introduction

We built the CADB dataset upon the existing Aesthetics and Attributes DataBase ([AADB](https://github.com/aimerykong/deepImageAestheticsAnalysis)). CADB dataset contains 9,497 images with each image rated by 5 individual raters who specialize in fine art for the overall composition quality, in which we provide a composition rating scale from 1 to 5, where a larger score indicates better composition. Some example images with annotations in CADB dataset are illustrated in the figure below, in which we show five composition scores provided by five raters in blue and the calculated composition mean score in red.

<img src='examples/annotation_example.jpg' align="center" width=1024>

## Download
Download dataset (~2GB) from 
[[Google Drive]](https://drive.google.com/file/d/1fpZoo5exRfoarqDvdLDpQVXVOKFW63vz/view?usp=sharing) | [[Baidu Cloud]](https://pan.baidu.com/s/1o3AktNB-kmOIanJtzEx98g)(access code: *rmnb*).

# Method Overview

## Motivation

<img src=''>

## SAMP-Net

# Results


# Code Usage
```bash
  # clone this repository
  git clone https://github.com/bcmi/Image-Composition-Assessment-with-SAMP.git
  cd Image-Composition-Assessment-with-SAMP/SAMPNet
  # download CADB data (~2GB), change the default dataset folder and gpu id in config.py.
  ```
## Training
```bash
   python train.py
   # track your experiments
   tensorboard --logdir=./experiments --bind_all
   ```
During training, the evaluation results of each epoch are recorded in a ``csv`` format file under the produced folder ``./experiments``.

## Testing
You can download pretrained model (~180MB) from [[Google Drive]](https://drive.google.com/file/d/1sIcYr5cQGbxm--tCGaASmN0xtE_r-QUg/view?usp=sharing) | [[Baidu Cloud]](https://pan.baidu.com/s/17EzhsbHqwA5aR8ty77fTvw)(access code: *5qgg*). 
```bash
   # place the pretrianed model in the folder ``pretrained_model`` and check the path in ``test.py``.
   # change the default gpu id in config.py
   python test.py
   ```

## Citation
```
@article{zhang2021image,
  title={Image Composition Assessment with Saliency-augmented Multi-pattern Pooling},
  author={Zhang, Bo and Niu, Li and Zhang, Liqing},
  journal={arXiv preprint arXiv:2104.03133},
  year={2021}
}
```
