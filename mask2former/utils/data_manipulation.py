from typing import Tuple, Union, List, Dict
import torch

import torch.nn.functional as F

from detectron2.structures.image_list import ImageList


def filter_pred_and_reduce(outputs, targets, indices):
    # outputs_occlusion = outputs["pred_occlusion"]
    outputs_occlusion = outputs

    preds_occ = []
    tgts_occ = []
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
            preds_occ.append(None)
            tgts_occ.append(None)
            continue

        tgt_occ = tgt_occ[reduced_tgt_pan_idx_bool][:, reduced_tgt_pan_idx_bool]

        # Reduce if not reduced before
        if tgt_occ.shape[0] != out_occ.shape[0]:
            pred_occ = reindex_geom_matrix_on_geom_gt(
                reduced_corresp_segs, reduced_triggered_queries, out_occ
            )
        else:
            pred_occ = out_occ
        preds_occ.append(pred_occ)
        tgts_occ.append(tgt_occ)
    return preds_occ, tgts_occ


def match_geom_pred_and_tgt(geom_pred, geom_tgt, tgt_pan_idx, matching):
    """
    Reindexes the geometric predictions and its corresponding target matrix
    so that every predicted position relates to the ground truth.
    """
    (
        reduced_triggered_queries,
        reduced_corresp_segs,
        reduced_tgt_pan_idx_bool,
    ) = reduce_pred_and_tgt(tgt_pan_idx, matching)

    geom_tgt_reindexed = geom_tgt[reduced_tgt_pan_idx_bool][:, reduced_tgt_pan_idx_bool]
    geom_pred_reindexed = reindex_geom_matrix_on_geom_gt(
        reduced_corresp_segs, reduced_triggered_queries, geom_pred
    )
    return geom_pred_reindexed, geom_tgt_reindexed


def reduce_pred_and_tgt(
    tgt_pan_idx: torch.Tensor, matching: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Check which predicted segments have an occlusion annotation
    and filters the triggered object queries indices along with the
    annotated occlusion annotations while keeping their relative
    indexing the same.
    Returns the reduced object queries, their reduced corresponding
    panoptic segment GT indices and a reduced tensor containing
    all the rows/cols of the occlusion annotations that have been predicted
    """
    triggered_queries, corresp_segs = matching
    triggered_queries = triggered_queries.to(tgt_pan_idx.device)
    corresp_segs = corresp_segs.to(tgt_pan_idx.device)
    # Reduction #
    # Are predicted segments in occ annotations ?
    ann_segs_bool = torch.isin(corresp_segs, tgt_pan_idx)
    reduced_corresp_segs = corresp_segs[ann_segs_bool]
    # Keep their object query relation
    reduced_triggered_queries = triggered_queries[ann_segs_bool]

    # Are the occ annotations predicted by the model ?
    ann_pan_idx_pred = torch.isin(tgt_pan_idx, corresp_segs)
    return reduced_triggered_queries, reduced_corresp_segs, ann_pan_idx_pred


def reindex_mask_preds_on_geom_gt(
    reduced_corresp_segs: torch.Tensor,
    reduced_triggered_queries: torch.Tensor,
    pred_masks: torch.Tensor,
):
    """
    Reindex mask pred on geom idx
    Since both reduced_corresp_segs and reduced_tgt_pan_idx contain the same segments indices,
    a trivial reindexing is the one that sorts the segments idx ascendingly.
    But reduced_tgt_pan_idx is always sorted ascendingly, so no change needed.
    """
    masks_sorting = torch.argsort(reduced_corresp_segs)
    sorted_reduced_oq = reduced_triggered_queries[masks_sorting]
    pred_masks = pred_masks[sorted_reduced_oq]
    return pred_masks


def reindex_geom_matrix_on_geom_gt(
    reduced_corresp_segs: torch.Tensor,
    reduced_triggered_queries: torch.Tensor,
    pred_matrix: torch.Tensor,
) -> torch.Tensor:
    geom_sorting = torch.argsort(reduced_corresp_segs)
    sorted_reduced_oq = reduced_triggered_queries[geom_sorting]
    pred_geom_matrix = pred_matrix[sorted_reduced_oq][:, sorted_reduced_oq]
    return pred_geom_matrix


def get_non_diag_values(matrix: torch.Tensor):
    # works for [N, N, C] matrices
    non_diag_idx = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return matrix[non_diag_idx]


def get_pairs_and_non_pairs_idx(
    tgt_geom: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Returns the position of all the positive samples and
    all the negative samples in an occlusion matrix in that order.
    """
    return torch.where(tgt_geom == 1), torch.where(tgt_geom == 0)


def get_matrix_num_rows_cols(triu_len: int):
    """Works with tril too, as long as diag is not included"""
    return int((1 + (8 * triu_len + 1) ** 0.5) / 2)


def get_any_geom_matrix(target: Dict[str, torch.Tensor]) -> torch.Tensor:
    geom_keys = [key for key in target.keys() if "matrix" in key]
    return target[geom_keys[0]]


def get_geom_matrix_keys(target: Dict[str, torch.Tensor]) -> List[str]:
    """Extracts every key in target if the key contains `matrix` in it."""
    return [k for k in target.keys() if "matrix" in k]


def resize_h_w(
    images: Union[ImageList, torch.Tensor], size: int, interp: str
) -> torch.Tensor:
    return F.interpolate(
        images.tensor if type(images) is ImageList else images,
        size=(size, size),
        mode=interp,
    )


def resize_images_and_masks(
    images: Union[ImageList, torch.Tensor], masks: torch.Tensor, size: int
):
    return resize_h_w(images, size=size, interp="bilinear"), resize_h_w(
        masks, size=size, interp="nearest"
    )


def pairwise_cos_sim(t: torch.Tensor):
    return F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(2), dim=-1)


def pairwise_euclidian_dist(t: torch.Tensor):
    B, D, _ = t.shape
    dist = torch.cdist(t, t)
    diag = torch.eye(D).repeat(B, 1, 1).bool()
    dist[diag] = 1
    return 1 / dist
