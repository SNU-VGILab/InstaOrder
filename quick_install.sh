#!/bin/bash

# torch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install opencv-python
conda install ipython shapely h5py scipy submitit scikit-image cython timm einops scikit-learn wandb -c anaconda -c conda-forge -y

# panopticapi
pip install git+https://github.com/cocodataset/panopticapi.git

# detectron2
cd .. && git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && git checkout 80307d2 && cd ..
python -m pip install -e detectron2

# msdeform attn
# TODO: change to IntaFormer here
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

# Utilities
conda install flake8 pyright black -c conda-forge -c anaconda -y
