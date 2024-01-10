from typing import List, Tuple, Dict

import numpy as np

import torch

from .data_manipulation import (
    get_any_geom_matrix,
    reduce_pred_and_tgt,
    reindex_mask_preds_on_geom_gt,
    get_geom_matrix_keys,
    get_pairs_and_non_pairs_idx,
)


OCC_MATRIX_STR: str = "occlusion_matrix"
DEPTH_MATRIX_STR: str = "depth_matrix"


def get_all_instances_pairs(
    pred_masks: torch.Tensor,  # pred_masks in our case
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    use_gt_masks: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Creates 2 sets of A over B and B over A for inference.
    """
    modal1 = []
    modal2 = []
    for b, (target, matching) in enumerate(zip(targets, indices)):
        tgt_pan_idx = target["tgt_pan_idx"]  # annotated pan segs in instaorder
        tgt_geom_matrix = get_any_geom_matrix(target)

        if use_gt_masks:
            tgt_masks = target["low_masks"]
            reindexed_pred_masks = tgt_masks
            tgt_geom = tgt_geom_matrix
        else:
            (
                reduced_triggered_queries,
                reduced_corresp_segs,
                reduced_tgt_pan_idx_bool,  # bool indexing
            ) = reduce_pred_and_tgt(tgt_pan_idx, matching)

            reindexed_pred_masks = reindex_mask_preds_on_geom_gt(
                reduced_corresp_segs, reduced_triggered_queries, pred_masks[b]
            )
            tgt_geom = tgt_geom_matrix[reduced_tgt_pan_idx_bool][
                :, reduced_tgt_pan_idx_bool
            ]
        # Creating all pairs of masks
        row_idx, col_idx = torch.triu_indices(
            tgt_geom.shape[0], tgt_geom.shape[1], offset=1
        )
        triu_masks = reindexed_pred_masks[row_idx]
        tril_masks = reindexed_pred_masks[col_idx]

        modal1.append(triu_masks.sigmoid().round().unsqueeze(1))
        modal2.append(tril_masks.sigmoid().round().unsqueeze(1))
    return modal1, modal2


def select_mask_pairs_and_gt(
    pred_masks: torch.Tensor,  # pred_masks in our case
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
):
    """
    2 phases : 1. Rduction : filtering predicted and annotated segments
               2. Reindexing : matching predictions to annotations
    """
    geom_matrix_keys = get_geom_matrix_keys(targets[0])

    B, _, H, W = pred_masks.shape
    modal1 = pred_masks.new_empty((B, 1, H, W))
    modal2 = pred_masks.new_empty((B, 1, H, W))
    gt_geom = {
        tgt_task: pred_masks.new_empty(
            (B, 2)
            if tgt_task == OCC_MATRIX_STR
            else (B, 1),  # NOTE: maybe just B for second arg
            dtype=torch.long,
        )
        for tgt_task in geom_matrix_keys
    }

    for b, (target, matching) in enumerate(zip(targets, indices)):
        tgt_pan_idx = target["tgt_pan_idx"]  # annotated panoptic segments in instaorder
        tgt_geom = {tgt_task: target[tgt_task] for tgt_task in geom_matrix_keys}
        (
            reduced_triggered_queries,
            reduced_corresp_segs,
            reduced_tgt_pan_idx,
        ) = reduce_pred_and_tgt(tgt_pan_idx, matching)

        reindexed_pred_masks = reindex_mask_preds_on_geom_gt(
            reduced_corresp_segs, reduced_triggered_queries, pred_masks[b]
        )
        tgt_geom = {
            tgt_task: tgt_geom[reduced_tgt_pan_idx][:, reduced_tgt_pan_idx]
            for tgt_task, tgt_geom in tgt_geom.items()
        }

        # Pair selection
        if DEPTH_MATRIX_STR in gt_geom.keys():
            sampled_indices = sample_pair_from_depth_matrix(tgt_geom[DEPTH_MATRIX_STR])
        else:  # means we are only training on occlusion
            pairs, non_pairs = get_pairs_and_non_pairs_idx(tgt_geom[OCC_MATRIX_STR])
            sampled_indices = sample_pair_from_occ_matrix(pairs, non_pairs)

        # prob_switch = np.random.rand()
        prob_switch = 0
        for task, tgt_matrix in tgt_geom.items():
            gt_relation = get_gt_relation(
                task, tgt_matrix, sampled_indices, device=pred_masks.device
            )
            if prob_switch > 0.5:
                gt_relation = switch_gt_relation_if_occ_or_depth(gt_relation, task)
            gt_geom[task][b] = gt_relation

        mask_pair = reindexed_pred_masks[sampled_indices, ...].sigmoid().round()
        if prob_switch > 0.5:
            modal1[b] = mask_pair[1]
            modal2[b] = mask_pair[0]
        else:
            modal1[b] = mask_pair[0]
            modal2[b] = mask_pair[1]
    return modal1.squeeze(0), modal2.squeeze(0), gt_geom


def sample_pair_from_occ_matrix(
    pairs: List[torch.Tensor],
    non_pairs: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # get pair 70% of time
    if len(pairs[0]) > 0 and len(non_pairs[0]) > 0:
        pair_or_non_pair = (
            pairs if np.random.rand() < 0.7 or len(non_pairs[0]) == 0 else non_pairs
        )
    elif len(pairs[0]) == 0:
        pair_or_non_pair = non_pairs
    else:
        pair_or_non_pair = pairs
    idx = np.random.choice(len(pair_or_non_pair[0]))
    seg_idx1 = pair_or_non_pair[0][idx]
    seg_idx2 = pair_or_non_pair[1][idx]
    indices = (seg_idx1, seg_idx2)
    return indices


def sample_pair_from_depth_matrix(
    tgt_depth_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # NOTE: think about case where there is no 1 or 2 in matrix
    # is it possible ? There must be at least one annotation in InstaOrder
    position_positive = torch.where(tgt_depth_matrix > 0)  # select 1s and 2s
    idx = np.random.choice(len(position_positive[0]))
    seg_idx1 = position_positive[0][idx]
    seg_idx2 = position_positive[1][idx]
    indices = (seg_idx1, seg_idx2)
    return indices


def get_gt_relation(
    task, tgt_geom: torch.Tensor, indices: Tuple[torch.Tensor, torch.Tensor], device
) -> torch.Tensor:
    seg_idx1, seg_idx2 = indices
    # NOTE: trying to invert
    gt_relation1 = tgt_geom[seg_idx2, seg_idx1]
    gt_relation2 = tgt_geom[seg_idx1, seg_idx2]
    if task == OCC_MATRIX_STR:
        geom_gt_pair = torch.tensor(
            [gt_relation1, gt_relation2],
            dtype=torch.long,
            device=device,
        )
        return geom_gt_pair
    elif task == DEPTH_MATRIX_STR:
        if gt_relation1 == -1:  # this should never happen... ask hyunmin why...
            gt_relation = -1
        elif gt_relation1 == 1 and gt_relation2 == 0:
            gt_relation = 0
        elif gt_relation1 == 2:
            gt_relation = 2
        else:
            raise RuntimeError  # TODO: get rid of that after understanding -1 case
        return torch.tensor([gt_relation], dtype=torch.long, device=device)
    else:  # count and overlap
        return gt_relation1.to(device).to(torch.long)


def switch_gt_relation_if_occ_or_depth(gt_relation, task: str):
    if task == OCC_MATRIX_STR:
        gt_relation = gt_relation[[1, 0]]
    if task == DEPTH_MATRIX_STR:
        gt_relation[0] = 1 if gt_relation == 0 else 1
    return gt_relation


def average_occlusion_preds(
    a_over_b: torch.Tensor, b_over_a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    a_over_b = torch.sigmoid(a_over_b)
    b_over_a = torch.sigmoid(b_over_a)
    prob_a_over_b = (a_over_b[:, 1] + b_over_a[:, 0]) / 2
    prob_b_over_a = (a_over_b[:, 0] + b_over_a[:, 1]) / 2
    return prob_a_over_b, prob_b_over_a


def average_depth_preds(
    a_before_b: torch.Tensor, b_before_a: torch.Tensor
) -> torch.Tensor:
    prob_1_closer_2 = (a_before_b[:, 0] + b_before_a[:, 1]) / 2
    prob_1_farther_2 = (a_before_b[:, 1] + b_before_a[:, 0]) / 2
    prob_1_equals_2 = (a_before_b[:, 2] + b_before_a[:, 2]) / 2
    return torch.argmax(torch.cat((prob_1_closer_2, prob_1_farther_2, prob_1_equals_2)))


def average_occlusion_depth_preds(
    occ_a_over_b: torch.Tensor,
    occ_b_over_a: torch.Tensor,
    depth_a_over_b: torch.Tensor,
    depth_b_over_a: torch.Tensor,
):
    return average_occlusion_preds(occ_a_over_b, occ_b_over_a), average_depth_preds(
        depth_a_over_b, depth_b_over_a
    )


# TODO: debug only. remove.
def visualize_masks(pred_masks, triu_masks, tril_masks):
    import matplotlib.pyplot as plt
    import os

    save_path = os.path.join(os.getcwd(), "dataset_viz", "out_masks")
    for i, mask in enumerate(pred_masks):
        fn = os.path.join(save_path, f"pred_masks_{i}.png")
        plt.imsave(fn, mask.sigmoid().round().cpu().numpy())
    for i, (triu, tril) in enumerate(zip(triu_masks, tril_masks)):
        fn = os.path.join(save_path, f"triu_{i}.png")
        plt.imsave(fn, triu.sigmoid().round().cpu().numpy())
        fn = os.path.join(save_path, f"tril_{i}.png")
        plt.imsave(fn, tril.sigmoid().round().cpu().numpy())


# TODO: debug only. remove.
def plot_masks(file_path: str, pred_masks: torch.Tensor, tgt_masks: torch.Tensor):
    import matplotlib.pyplot as plt

    num_cols = pred_masks.shape[0]

    fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(3 * num_cols, 3))
    for i, (pred, tgt) in enumerate(zip(pred_masks, tgt_masks)):
        axes[0, i].imshow(pred.sigmoid().round().int().cpu().numpy())
        # axes[0, i].axis("off")
        axes[1, i].imshow(tgt.sigmoid().round().int().cpu().numpy())
        # axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Pred")
    axes[1, 0].set_ylabel("Target")
    # Not sure that it works...
    for ax in axes.flat:
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{file_path}")
    plt.close(fig)
