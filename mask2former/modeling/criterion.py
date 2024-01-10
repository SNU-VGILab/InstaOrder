# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils.data_manipulation import (
    reduce_pred_and_tgt,
    reindex_geom_matrix_on_geom_gt,
    get_non_diag_values,
)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


class BalancedSoftmax(nn.modules.loss._Loss):
    def __init__(self, samples_per_class):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = torch.tensor(samples_per_class)

    def balanced_softmax_loss(self, logits, labels, sample_per_class, reduction="mean"):
        spc = sample_per_class.to(logits.device)  # (num_classes, )
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)  # (batch_size, num_classes)
        logits = logits + spc.log()  # (batch_size, num_classes)
        loss = F.cross_entropy(
            input=logits, target=labels.long(), reduction=reduction
        )  # (batch_size, )
        return loss

    def forward(self, input, label, reduction="mean"):
        return self.balanced_softmax_loss(
            input, label, self.sample_per_class, reduction
        )

    def __str__(self):
        return f"BalancedSoftmax(sample_per_class{self.sample_per_class})"


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses: List[str],
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        # InstaOrder
        pairwise_training: bool,
        occ_samples_per_class: List[int],
        depth_samples_per_class: List[int],
        depth_loss: str,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # InstaOrder
        self.pairwise_training = pairwise_training
        self.depth_loss_type = depth_loss
        self.geom_losses = {
            "occlusion": BalancedSoftmax(occ_samples_per_class),
            "depth": F.cross_entropy
            if depth_loss == "ce"
            else BalancedSoftmax(depth_samples_per_class),
        }

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_pairwise_occlusion(self, outputs, targets, indices, _):
        assert "pred_occlusion" in outputs
        pred_a_over_b, pred_b_over_a = outputs["pred_occlusion"]
        tgt_a_over_b = torch.stack(
            [target["occlusion_matrix"] for target in targets]
        ).float()
        tgt_b_over_a = tgt_a_over_b[:, [1, 0]]
        loss_a_over_b = F.binary_cross_entropy_with_logits(pred_a_over_b, tgt_a_over_b)
        loss_b_over_a = F.binary_cross_entropy_with_logits(pred_b_over_a, tgt_b_over_a)
        loss = loss_a_over_b + loss_b_over_a
        return {"loss_occlusion": loss}

    def loss_pairwise_depth(self, outputs, targets, indices, _):
        return NotImplementedError

    def loss_occlusion(self, outputs, targets, indices, _):
        assert "pred_occlusion" in outputs

        outputs_occlusion = outputs["pred_occlusion"]
        sample_losses = []

        for out_occ, target, match in zip(outputs_occlusion, targets, indices):
            tgt_occ = target["occlusion_matrix"].to(out_occ.device)
            tgt_pan_idx = target["tgt_pan_idx"]
            (
                reduced_triggered_queries,
                reduced_corresp_segs,
                reduced_tgt_pan_idx_bool,
            ) = reduce_pred_and_tgt(tgt_pan_idx, match)

            # Not enough predictions to infer geometric orderings
            if reduced_tgt_pan_idx_bool.sum() < 2:
                continue

            tgt_occ = tgt_occ[reduced_tgt_pan_idx_bool][:, reduced_tgt_pan_idx_bool]

            # Reduce if not reduced before
            if tgt_occ.shape[0] != out_occ.shape[0]:
                pred_occ = reindex_geom_matrix_on_geom_gt(
                    reduced_corresp_segs, reduced_triggered_queries, out_occ
                )
            else:
                pred_occ = out_occ

            tgt_occ = get_non_diag_values(tgt_occ)
            pred_occ = get_non_diag_values(pred_occ)

            sample_loss = self.geom_losses["occlusion"](pred_occ, tgt_occ.float())
            sample_losses.append(sample_loss)
        loss = torch.stack(sample_losses).sum()

        losses = {"loss_occlusion": loss}
        return losses

    def loss_depth(self, outputs, targets, indices, _):
        assert "pred_depth" in outputs
        outputs_depth = outputs["pred_depth"]
        sample_losses = []

        for out_depth, target, match in zip(outputs_depth, targets, indices):
            tgt_depth = target["depth_matrix"].to(out_depth.device)
            tgt_pan_idx = target["tgt_pan_idx"]
            (
                reduced_triggered_queries,
                reduced_corresp_segs,
                reduced_tgt_pan_idx_bool,
            ) = reduce_pred_and_tgt(tgt_pan_idx, match)

            # Not enough predictions to infer geometric orderings
            if reduced_tgt_pan_idx_bool.sum() < 2:
                continue

            tgt_depth = tgt_depth[reduced_tgt_pan_idx_bool][:, reduced_tgt_pan_idx_bool]

            # Reduce if not reduced before
            if tgt_depth.shape[0] != out_depth.shape[0]:
                pred_depth = reindex_geom_matrix_on_geom_gt(
                    reduced_corresp_segs, reduced_triggered_queries, out_depth
                )
            else:
                pred_depth = out_depth

            # Prediction ranges from [[-1, 2]], so add one to make softmax happy
            # print(f"tgt\n{tgt_depth}\npred\n{pred_depth.argmax(-1)}")
            tgt_depth = get_non_diag_values(tgt_depth)
            pred_depth = get_non_diag_values(pred_depth)

            if self.depth_loss_type == "ce":
                tgt_depth = tgt_depth.long()
            else:
                tgt_depth = tgt_depth.float()
            sample_loss = self.geom_losses["depth"](pred_depth, tgt_depth)
            sample_losses.append(sample_loss)
        loss = torch.stack(sample_losses).sum()

        losses = {"loss_depth": loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following iNdices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "occlusion": self.loss_pairwise_occlusion
            if self.pairwise_training
            else self.loss_occlusion,
            "depth": self.loss_pairwise_depth
            if self.pairwise_training
            else self.loss_depth,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if (
                        f"pred_{loss}" in aux_outputs.keys()
                    ):  # different qtty of aux losses
                        l_dict = self.get_loss(
                            loss, aux_outputs, targets, indices, num_masks
                        )
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
