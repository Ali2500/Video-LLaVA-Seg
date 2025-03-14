# Video-LLaVA-Seg

[Ali Athar](https://www.aliathar.net/), [Xueqing Deng](https://sites.google.com/view/xueqingdeng7/home), [Liang-Chieh Chen](http://liangchiehchen.com/)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://ali2500.github.io/vicas-project/)
[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://huggingface.co/datasets/Ali2500/ViCaS)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.09754)
[![Full Paper](https://img.shields.io/badge/Full_Paper-Read-0000FF.svg)](https://arxiv.org/pdf/2412.09754)

This is the official baseline implementation for the [ViCaS dataset](https://ali2500.github.io/vicas-project/) (CVPR'25). The main project GitHub repo is [here](https://github.com/Ali2500/ViCaS/tree/main).

The trained model is uploaded to [HuggingFace](https://huggingface.co/fun-research/Video-LLaVA-Seg).

## Environment

Create a new conda environment with Python 3.9.2 and activate it:

```bash
conda create -n videollavaseg python==3.9.2
conda activate videollavaseg
```

Install the correct version of PyTorch. We used CUDA 12.1 in our setup:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

Install Flash Attention:

```bash
pip install flash-attn==2.6.3 --no-build-isolation
```

## Dataset Structure

Create a `datasets` directory and place the ViCaS dataset in it. For instructions on downloading the dataset, refer to the [main dataset GitHub repo](https://github.com/Ali2500/ViCaS/tree/main?tab=readme-ov-file#arrow_double_down-dataset-download). The repo folder should then look like this:

```
$REPO_DIR
├── llava                      
│   ├── ...
├── datasets
│   ├── ViCaS
│   │   └── splits
│   │   │   ├── v0.1
│   │   │   ├── v1.0
│   │   └── annotations
│   │   │   ├── v0.1
│   │   │   ├── v1.0
│   │   └── videos
│   │   └── video_frames
```

## Inference

Run the following command:

```bash
python llava/inference/main.py -i /path/to/model/directory -o /path/to/output --dataset_split {val,test}
```

If you're on a multi-GPU setup then you can parallelize the inference by running the inference script with the same arguments:

```bash
bash scripts/infer.sh -i /path/to/model/directory -o /path/to/output --dataset_split {val,test}
```

## ⚠️ Terms of use
* This model cannot be used for commercial purposes. It has been created for research purposes only.
* This is not an official ByteDance product.

## BibTeX

```
@article{athar2024vicas,
author = {Ali Athar, Xueqing Deng, Liang-Chieh Chen},
title = {ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation},
journal = {CVPR},
year = {2025}
}
```

