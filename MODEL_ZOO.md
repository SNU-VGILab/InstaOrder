# Mask2Former Model Zoo and Baselines

> This model zoo file is a short version of [Mask2Former's model zoo file](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md).

## Introduction

This file documents a collection of models reported in our paper.
All numbers were obtained on [Big Basin](https://engineering.fb.com/data-center-engineering/introducing-big-basin-our-next-generation-ai-hardware/)
servers with 8 NVIDIA V100 GPUs & NVLink (except Swin-L models are trained with 16 NVIDIA V100 GPUs).

#### How to Read the Tables

- The "Name" column contains a link to the config file. Running `train_net.py --num-gpus 8` with this config file
  will reproduce the model (except Swin-L models are trained with 16 NVIDIA V100 GPUs with distributed training on two nodes).
- The _model id_ column is provided for ease of reference.
  To check downloaded file integrity, any model on this page contains its md5 prefix in its file name.

#### Third-party ImageNet Pretrained Models

Our paper also uses ImageNet pretrained models that are not part of Detectron2, please refer to [tools](https://github.com/facebookresearch/MaskFormer/tree/master/tools) to get those pretrained models.

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

## COCO Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">50</td>
<td align="center">53.2</td>
<td align="center">43.3</td>
<td align="center">63.2</td>
<td align="center">48558700_1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_tiny_bs16_50ep/model_final_9fd0ae.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">50</td>
<td align="center">54.6</td>
<td align="center">44.7</td>
<td align="center">64.2</td>
<td align="center">48558700_3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_small_bs16_50ep/model_final_a407fd.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B</td>
<td align="center">50</td>
<td align="center">55.1</td>
<td align="center">45.2</td>
<td align="center">65.1</td>
<td align="center">48558700_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_384_bs16_50ep/model_final_9d7f02.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">50</td>
<td align="center">56.4</td>
<td align="center">46.3</td>
<td align="center">67.1</td>
<td align="center">48558700_7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_100ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">100</td>
<td align="center">57.8</td>
<td align="center">48.6</td>
<td align="center">67.4</td>
<td align="center">47429163_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl">model</a></td>
</tr>
</tbody></table>
