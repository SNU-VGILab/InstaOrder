# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, List, Union, Dict
from itertools import zip_longest
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

# InstaOrder
from .modeling.instaorder_predictors import build_geometric_predictor
from .utils.pairwise import (
    select_mask_pairs_and_gt,
    get_all_instances_pairs,
    average_occlusion_preds,
    average_depth_preds,
    average_occlusion_depth_preds,
)
from .utils.data_manipulation import (
    filter_pred_and_reduce,
    reduce_pred_and_tgt,
    reindex_mask_preds_on_geom_gt,
    match_geom_pred_and_tgt,
    resize_h_w,
    get_matrix_num_rows_cols,
)


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        geometric_predictor: nn.Module,  # InstaOrder
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        occ_pred_input_size: int,  # InstaOrder
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # InstaOrder
        pairwise: bool,
        geometric_task: str,
        # evaluation InstaOrder
        eval_occ: bool,
        eval_depth: bool,
        # inference InstaOrder
        occlusion_on: bool,
        depth_on: bool,
        use_gt_masks: bool,
        # Freezing modules
        freeze_backbone: bool,
        freeze_sem_seg_head: bool,
        filter_triggered_queries: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.geometric_predictor = geometric_predictor  # InstaOrder
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.occ_pred_input_size = occ_pred_input_size  # InstaOrder
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        # InstaOrder
        self.occlusion_on = occlusion_on
        self.depth_on = depth_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # InstaOrder
        self.pairwise = pairwise
        self.geometric_task = geometric_task
        if geometric_task != "":
            assert geometric_task in [
                "o",
                "d",
                "od",
            ], "Geometric task must be one of {'o', 'd', 'od', ''}."

            self.forward_geometric = {
                "o": self.forward_order_net,
                "d": self.forward_depth_net,
                "od": self.forward_order_depth_net,
            }[geometric_task]
            self.post_process_pairwise_infer = {
                "o": average_occlusion_preds,
                "d": average_depth_preds,
                "od": average_occlusion_depth_preds,
            }[geometric_task]

        # occlusion evaluation
        self.eval_occ = eval_occ
        self.eval_depth = eval_depth
        self.use_gt_masks = use_gt_masks  # debugging purposes
        self.filter_triggered_queries = filter_triggered_queries

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_sem_seg_head:
            for param in self.sem_seg_head.parameters():
                param.requires_grad = False

        # Trainable adapters
        if hasattr(self.sem_seg_head.predictor, "transformer_adapter_layers"):
            for (
                param
            ) in self.sem_seg_head.predictor.transformer_adapter_layers.parameters():
                param.requires_grad = True
            for module in self.sem_seg_head.pixel_decoder.transformer.encoder.layers:
                for param in module.adapter.parameters():
                    param.requires_grad = True

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        # InstaOrder
        pairwise = cfg.MODEL.GEOMETRIC_PREDICTOR.PAIRWISE
        geometric_predictor = (
            build_geometric_predictor(cfg)
            if pairwise or cfg.MODEL.GEOMETRIC_PREDICTOR.NAME != ""
            else None
        )

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        occlusion_weight = cfg.MODEL.GEOMETRIC_PREDICTOR.OCCLUSION_WEIGHT
        depth_weight = cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_occlusion": occlusion_weight,
            "loss_depth": depth_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        geometric_task = cfg.MODEL.GEOMETRIC_PREDICTOR.TASK
        if geometric_task == "o" or geometric_task in ["o", "od"]:
            losses.append("occlusion")
        if geometric_task == "d" or geometric_task in ["d", "od"]:
            losses.append("depth")

        for loss in cfg.MODEL.REMOVE_LOSSES:
            losses.remove(loss)

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            # InstaOrder
            pairwise_training=cfg.MODEL.GEOMETRIC_PREDICTOR.PAIRWISE,
            occ_samples_per_class=cfg.INPUT.OCC_SAMPLES_PER_CLASS,
            depth_samples_per_class=cfg.INPUT.DEPTH_SAMPLES_PER_CLASS,
            depth_loss=cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_LOSS,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "geometric_predictor": geometric_predictor,  # InstaOrder
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "occ_pred_input_size": cfg.MODEL.GEOMETRIC_PREDICTOR.INPUT_SIZE,  # InstaOrder
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # InstaOrder pairwise training
            "pairwise": cfg.MODEL.GEOMETRIC_PREDICTOR.PAIRWISE,
            "geometric_task": geometric_task,
            # occlusion evaluation
            "eval_occ": cfg.TEST.OCCLUSION_EVALUATION,
            "eval_depth": cfg.TEST.DEPTH_EVALUATION,
            # Inference
            "occlusion_on": cfg.MODEL.GEOMETRIC_PREDICTOR.OCCLUSION_ON,
            "depth_on": cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_ON,
            "use_gt_masks": cfg.MODEL.GEOMETRIC_PREDICTOR.USE_GT_MASKS,
            # Freezing modules
            "freeze_backbone": cfg.MODEL.BACKBONE.FREEZE,
            "freeze_sem_seg_head": cfg.MODEL.SEM_SEG_HEAD.FREEZE,
            "filter_triggered_queries": cfg.MODEL.GEOMETRIC_PREDICTOR.FILTER_TRIGGERED_QUERIES,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def prepare_multi_task_targets(self, batched_inputs, images):
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            if "occlusion_matrix" in batched_inputs[0]:
                targets = self.add_occ_to_targets(batched_inputs, targets)
            if "depth_matrix" in batched_inputs[0]:
                targets = self.add_depth_to_targets(batched_inputs, targets)
        else:
            targets = None
        return targets

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs, mask_embed, mask_features = self.sem_seg_head(features)

        if self.training:
            targets = self.prepare_multi_task_targets(batched_inputs, images)
            if self.geometric_task != "":
                # TODO: finish pairwise training routine
                matching = self.criterion.matcher(outputs, targets)
                if self.pairwise:
                    pred_masks = outputs["pred_masks"]
                    low_images, low_masks = self.prepare_for_instaordernet(
                        batched_inputs, pred_masks, targets
                    )
                    low_images = torch.stack(low_images)
                    modal1, modal2, gt_pair = select_mask_pairs_and_gt(
                        low_masks, targets, matching
                    )
                    targets = self.add_pairwise_gt_to_targets(gt_pair, targets)
                    geom_preds = self.forward_geometric(modal1, modal2, low_images)
                    if "o" in self.geometric_task:
                        # Storing a tuple of (a_over_b, b_over_a)
                        outputs["pred_occlusion"] = geom_preds
                    if "d" in self.geometric_task:
                        raise NotImplementedError
                elif self.geometric_predictor is not None:
                    outputs = self.forward_geom_pred(
                        mask_embed, mask_features, matching, targets, outputs
                    )
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            geom_ordering_results = []

            if self.eval_occ or self.eval_depth:  # InstaOrder
                # Perfect matching for deterministic occlusion evaluation
                targets = self.prepare_multi_task_targets(batched_inputs, images)
                matching = self.criterion.matcher.full_mask_matching(outputs, targets)

                if self.pairwise:
                    low_images, masks = self.prepare_for_instaordernet(
                        batched_inputs, mask_pred_results, targets
                    )
                    (
                        stacked_masks1,
                        stacked_masks2,
                    ) = get_all_instances_pairs(
                        masks, targets, matching, self.use_gt_masks
                    )
                    for masks1, masks2, image in zip(
                        stacked_masks1, stacked_masks2, low_images
                    ):
                        image = image.unsqueeze(0)
                        sample_pred = defaultdict(list)
                        for m1, m2 in zip(masks1, masks2):
                            geom_preds = self.forward_geometric(m1, m2, image)
                            preds = self.post_process_pairwise_infer(*geom_preds)
                            if self.geometric_task == "o":
                                sample_pred["occlusion"].append([*preds])
                            if self.geometric_task == "d":
                                sample_pred["depth"].append(preds)
                            if self.geometric_task == "od":
                                occ, depth = preds
                                sample_pred["occlusion"].append([*occ])
                                sample_pred["depth"].append(depth)
                        geom_ordering_results.append(sample_pred)
                else:
                    if self.geometric_predictor is not None:
                        outputs = self.forward_geom_pred(
                            mask_embed, mask_features, matching, targets, outputs
                        )
                    occlusion_results = outputs.get("pred_occlusion")
                    depth_results = outputs.get("pred_depth")
                    for b, (tgt, match) in enumerate(zip(targets, matching)):
                        sample_pred = {}
                        tgt_occ = tgt.get("occlusion_matrix", None)
                        tgt_depth = tgt.get("depth_matrix", None)
                        tgt_pan_idx = tgt["tgt_pan_idx"]
                        if tgt_occ is not None:
                            pred_occ = occlusion_results[b]
                            if pred_occ.shape[:2] != tgt_occ.shape:
                                pred_occ, tgt_occ = match_geom_pred_and_tgt(
                                    pred_occ, tgt_occ, tgt_pan_idx, match
                                )
                            # pred_occ = pred_occ.sigmoid().round().int()
                            # TODO: look dynamically for which more we are using
                            pred_occ = pred_occ.argmax(-1).int()
                            sample_pred["occlusion"] = pred_occ
                        if tgt_depth is not None:
                            pred_depth = depth_results[b]
                            # pred_depth, tgt_depth = match_geom_pred_and_tgt(
                            #     pred_depth, tgt_depth, tgt_pan_idx, match
                            # )
                            # TODO: check the output preds and maybe modify them
                            pred_depth = pred_depth.argmax(-1).int()
                            sample_pred["depth"] = pred_depth
                        geom_ordering_results.append(sample_pred)
            else:
                # inference w/o GT
                if self.panoptic_on and self.geometric_predictor is not None:
                    # inference panoptique garder en memoire les label
                    # et ensuite seulement faire l'inference geometrique
                    processed_results = self.run_blind_inference(
                        mask_embed,
                        mask_features,
                        mask_cls_results,
                        mask_pred_results,
                        batched_inputs,
                        images,
                    )
                    return processed_results

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for (
                mask_cls_result,
                mask_pred_result,
                geom_ordering_result,
                input_per_image,
                image_size,
            ) in zip_longest(
                mask_cls_results,
                mask_pred_results,
                geom_ordering_results,
                batched_inputs,
                images.image_sizes,
                fillvalue=None,
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width
                        )
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

                # geometric ordering inference
                if self.occlusion_on and geom_ordering_results is not None:
                    if (
                        self.pairwise
                        and geom_ordering_result.get("occlusion") is not None
                    ):
                        occlusion_r = retry_if_cuda_oom(self.occ_pairwise_inference)(
                            geom_ordering_result["occlusion"]
                        )
                    else:  # parallel
                        # evaluation
                        if self.eval_occ:
                            occlusion_r = geom_ordering_result["occlusion"]
                        else:  # pure inference
                            raise NotImplementedError
                    processed_results[-1]["occlusion"] = occlusion_r
                if self.depth_on and geom_ordering_results is not None:
                    if self.pairwise and geom_ordering_result.get("depth") is not None:
                        depth_r = retry_if_cuda_oom(self.depth_pairwise_inference)(
                            geom_ordering_result["depth"]
                        )
                    else:  # parallel
                        if self.eval_depth:
                            depth_r = geom_ordering_result["depth"]
                        else:  # pure inference
                            pass
                    processed_results[-1]["depth"] = depth_r
            return processed_results

    def filter(
        self,
        src: torch.Tensor,
        filter: List[Tuple[torch.Tensor, torch.Tensor]],
        targets=None,
    ) -> List[torch.Tensor]:
        """
        Assumes `src` input of shape [B, N, ...], where B is the batch size
        and N is the number of object queries from which we will filter
        """
        if targets is not None:  # Training or eval
            ret = []
            for pred, target, matching in zip(src, targets, filter):
                # always filter and reindex
                (
                    reduced_triggered_queries,
                    reduced_corresp_segs,
                    _,
                ) = reduce_pred_and_tgt(target["tgt_pan_idx"], matching)
                # NOTE: forgot to reindex here...!
                reduced_masks = reindex_mask_preds_on_geom_gt(
                    reduced_corresp_segs, reduced_triggered_queries, pred
                )
                ret.append(reduced_masks)
        else:  # Inference w/o GT
            ret = [s[f] for s, f in zip(src, filter)]
        return ret

    def forward_geom_pred(self, mask_embed, mask_features, matching, targets, outputs):
        if self.training or self.eval_occ or self.eval_depth:
            mask_embed = self.filter(mask_embed, matching, targets)
            masks = self.filter(
                outputs["pred_masks"].sigmoid().round().int(), matching, targets
            )
            mask_features = [
                mask.unsqueeze(1) * features
                for mask, features in zip(masks, mask_features)
            ]
        else:
            mask_embed = self.filter(mask_embed, matching, targets)
            masks = self.filter(outputs.sigmoid().round().int(), matching, targets)
            mask_features = [
                mask.unsqueeze(1) * features
                for mask, features in zip(masks, mask_features)
            ]
        if self.geometric_predictor is not None:
            geom_preds = self.geometric_predictor(
                mask_embed, mask_features
            )  # {"task": [L, B, n, n, 2]}
            if self.training or self.eval_occ or self.eval_depth:
                outputs = self.add_geom_preds_to_output(geom_preds, outputs)
            else:
                outputs = self.extract_geom_preds(geom_preds)
        return outputs

    def extract_geom_preds(self, geom_preds):
        return {task: pred[-1] for task, pred in geom_preds.items()}

    def filter_geom_preds(self, geom_preds: defaultdict, targets: list, matching: list):
        for task, out_layer in geom_preds.items():
            for layer, pred in enumerate(out_layer):
                filtered_pred, _ = filter_pred_and_reduce(pred, targets, matching)
                geom_preds[task][layer] = filtered_pred
        return geom_preds  # ["task": [L, B, n, n, 2]]

    def prepare_for_instaordernet(
        self,
        batched_inputs: List[dict],
        mask_pred_results: torch.Tensor,
        targets: List[dict],
    ):
        low_images = [x["low_image"].to(self.device) for x in batched_inputs]
        if self.use_gt_masks:  # COCO masks
            low_masks = [x["low_masks"].to(self.device) for x in batched_inputs]
            for i, target in enumerate(targets):
                target["low_masks"] = low_masks[i].squeeze(0)

        masks = resize_h_w(
            mask_pred_results,
            size=self.occ_pred_input_size,
            interp="bilinear",  # changing from nearest to bilinear ?
        )
        return low_images, masks

    def forward_order_net(
        self, modal1: torch.Tensor, modal2: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards masks and image into an InstaOrderNet.
        modal1.shape = [B, H, W]
        modal2.shape = [B, H, W]
        image.shape = [B, C, H, W]
        """
        if len(modal1.shape) < 4:
            modal1 = modal1.unsqueeze(1)  # [B, 1, H, W]
            modal2 = modal2.unsqueeze(1)  # [B, 1, H, W]
        # NOTE : assume sigmoid is done downstream (train time loss, inference in pairwise.py)
        a_over_b = self.geometric_predictor(torch.cat([modal1, modal2, image], dim=1))
        b_over_a = self.geometric_predictor(torch.cat([modal2, modal1, image], dim=1))
        return a_over_b, b_over_a

    def forward_depth_net(
        self, modal1: torch.Tensor, modal2: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        modal1 = modal1.unsqueeze(1)
        modal2 = modal2.unsqueeze(1)
        a_over_b = nn.functional.softmax(
            self.geometric_predictor(torch.cat([modal1, modal2, image], dim=1)), dim=1
        )
        b_over_a = nn.functional.softmax(
            self.geometric_predictor(torch.cat([modal2, modal1, image], dim=1)), dim=1
        )
        return a_over_b, b_over_a

    def forward_order_depth_net(
        self, modal1: torch.Tensor, modal2: torch.Tensor, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        modal1 = modal1.unsqueeze(1)
        modal2 = modal2.unsqueeze(1)
        occ1, depth1 = self.geometric_predictor(
            torch.cat([modal1, modal2, image], dim=1)
        )
        occ2, depth2 = self.geometric_predictor(
            torch.cat([modal2, modal1, image], dim=1)
        )
        occ1 = torch.sigmoid(occ1)
        occ2 = torch.sigmoid(occ2)
        depth1 = nn.functional.softmax(depth1, dim=1)
        depth2 = nn.functional.softmax(depth2, dim=1)
        return occ1, occ2, depth1, depth2

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def add_pairwise_gt_to_targets(
        self, pairwise_gt_dict: Dict[str, torch.Tensor], targets
    ):
        for task, pairwise_gt in pairwise_gt_dict.items():
            for s, pair_gt in enumerate(pairwise_gt):
                targets[s][task] = pair_gt
        return targets

    def add_occ_to_targets(self, batched_inputs, targets):
        for sample, target in zip(batched_inputs, targets):
            target["occlusion_matrix"] = sample["occlusion_matrix"]
            target["tgt_pan_idx"] = sample["tgt_pan_idx"]
        return targets

    def add_depth_to_targets(self, batched_inputs, targets):
        for sample, target in zip(batched_inputs, targets):
            target["depth_matrix"] = sample["depth_matrix"]
            target["count_matrix"] = sample["count_matrix"]
            target["overlap_matrix"] = sample["overlap_matrix"]
            target["tgt_pan_idx"] = sample["tgt_pan_idx"]
        return targets

    def add_geom_preds_to_output(self, geom_preds, outputs):
        # TODO: redo this part (first condition)
        if type(geom_preds) is list:
            tasks = geom_preds[0].keys()
            for task in tasks:
                outputs[task] = []
            for geom_pred in geom_preds:  # over batch
                for task, pred in geom_pred.items():  # over task
                    for layer, p in enumerate(pred[:-1]):
                        if task not in outputs["aux_outputs"][layer]:
                            outputs["aux_outputs"][layer][task] = []
                        outputs["aux_outputs"][layer][task].append(p)
                    outputs[task].append(pred[-1])
        elif hasattr(geom_preds, "keys"):
            # [L, B, (n, n, 2)]
            outputs_occlusion = geom_preds.get("pred_occlusion", None)
            outputs_depth = geom_preds.get("pred_depth", None)
            if outputs_occlusion is not None:
                self.add_geom_pred_to_output(
                    outputs_occlusion, outputs, key="pred_occlusion"
                )
            if outputs_depth is not None:
                self.add_geom_pred_to_output(outputs_depth, outputs, key="pred_depth")
        return outputs

    def add_geom_pred_to_output(
        self, geom_pred: List[torch.Tensor], outputs: dict, key: str
    ) -> dict:
        # [L, B, (n, n, 2)]
        outputs[key] = geom_pred[-1]
        for r, out_occ in zip(outputs["aux_outputs"], geom_pred[:-1]):
            r[key] = out_occ
        return outputs

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(
        self, mask_cls, mask_pred, return_selected_seg_idx: bool = False
    ):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            if return_selected_seg_idx:
                return panoptic_seg, segments_info, keep
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = (
                    lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def occ_pairwise_inference(
        self, binary_occ_relations: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Given a set of pairwise occlusion prediction results, generates its
        occlusion matrix.
        """
        num_instances = get_matrix_num_rows_cols(len(binary_occ_relations))
        bin_occ_relations = torch.as_tensor(binary_occ_relations)
        bin_occ_relations = (bin_occ_relations > 0.5).int()
        is_a_over_b = bin_occ_relations[:, 0]
        is_b_over_a = bin_occ_relations[:, 1]

        triu_row, triu_col = torch.triu_indices(
            num_instances, num_instances, offset=1, device=is_a_over_b.device
        )
        occlusion_r = torch.full(
            (num_instances, num_instances),
            0,
            dtype=torch.int,
            device=is_a_over_b.device,
        )

        occlusion_r[triu_row, triu_col] = is_a_over_b
        occlusion_r[triu_col, triu_row] = is_b_over_a
        return occlusion_r

    def depth_pairwise_inference(
        self, depth_relations: List[torch.Tensor]
    ) -> torch.Tensor:
        num_instances = get_matrix_num_rows_cols(len(depth_relations))
        depth_relations = torch.tensor(depth_relations, dtype=torch.int)
        depth_r = torch.zeros((num_instances, num_instances), dtype=torch.int)
        triu_row, triu_col = torch.triu_indices(num_instances, num_instances, offset=1)

        zeros = depth_relations == 0
        ones = depth_relations == 1
        twos = depth_relations == 2
        zeros = torch.where(depth_relations == 0)
        ones = torch.where(depth_relations == 1)
        twos = torch.where(depth_relations == 2)
        triu_values = torch.zeros_like(depth_relations)
        tril_values = torch.zeros_like(depth_relations)
        triu_values[zeros] = 1
        triu_values[ones] = 0
        triu_values[twos] = 2
        tril_values[zeros] = 0
        tril_values[ones] = 1
        tril_values[twos] = 2
        depth_r[triu_row, triu_col] = triu_values
        depth_r[triu_col, triu_row] = tril_values
        return depth_r

    def run_blind_inference(
        self,
        mask_embed,
        mask_features,
        mask_cls_results,
        mask_pred_results,
        batched_inputs,
        images,
    ):
        resized_mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        processed_results = []
        for (
            resized_mask_pred_result,
            mask_cls_result,
            input_per_image,
            image_size,
        ) in zip(
            resized_mask_pred_results,
            mask_cls_results,
            batched_inputs,
            images.image_sizes,
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            processed_results.append({})
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    resized_mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # panoptic segmentation inference
            selected_indices = []
            if self.panoptic_on:
                masks, classes, selected_idx = retry_if_cuda_oom(
                    self.panoptic_inference
                )(mask_cls_result, mask_pred_result, return_selected_seg_idx=True)
                processed_results[-1]["panoptic_seg"] = (masks, classes)
                selected_indices.append(selected_idx)

        # Create list and attach results
        geom_ordering_results = self.forward_geom_pred(
            mask_embed, mask_features, selected_indices, None, mask_pred_results
        )

        if self.occlusion_on:
            occlusion_r = geom_ordering_results["pred_occlusion"]
            occlusion_r = [res.argmax(-1) for res in occlusion_r]
            # processed_results["occlusion"] = occlusion_r
            for res, geom in zip(processed_results, occlusion_r):
                res["occlusion"] = geom.fill_diagonal_(-1)
        if self.depth_on:
            depth_r = geom_ordering_results["pred_depth"]
            depth_r = [res.argmax(-1) for res in depth_r]
            for res, geom in zip(processed_results, depth_r):
                res["depth"] = geom.fill_diagonal_(-1)
        return processed_results
