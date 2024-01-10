from typing import List, Dict
from collections import defaultdict
from itertools import chain

import numpy as np

import torch
from detectron2.utils.logger import setup_logger
from detectron2.config import CfgNode, configurable
from detectron2.engine.defaults import DatasetEvaluator

from detectron2.utils import comm

logger = setup_logger(name="d2.evaluation.evaluator.depth")


class DepthEvaluator(DatasetEvaluator):
    @configurable
    def __init__(self, sample_verbose: bool, multi_scale_evaluation: bool = False):
        self.sample_verbose = sample_verbose
        self.multi_scale_evaluation = multi_scale_evaluation

    @classmethod
    def from_config(cls, cfg: CfgNode):
        return {
            "sample_verbose": cfg.TEST.SAMPLE_VERBOSE,
            "multi_scale_evaluation": cfg.TEST.MULTI_SCALE_EVALUATION,
        }

    def reset(self) -> None:
        self._whdr = defaultdict(list)
        self._n_instances = []

        # self._fn = []

    def calculate_whdr(self, order_matrix, gt_order_matrix, score_matrix, mask):
        if mask.sum() == 0:
            return -1
        whdr = (
            (gt_order_matrix[mask] != order_matrix[mask]) * score_matrix[mask]
        ).sum() / score_matrix[mask].sum()
        return whdr * 100

    def process(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        outputs: List[Dict[str, torch.Tensor]],
    ) -> None:
        def extract_triu_without_diag(M: torch.Tensor) -> torch.Tensor:
            # return M[torch.triu_indices(M.shape[0], M.shape[1], offset=1)]
            mask = torch.triu(torch.ones_like(M, dtype=bool), diagonal=1)
            return M[mask]

        for input_, output in zip(inputs, outputs):
            tgt_depth = extract_triu_without_diag(input_["depth_matrix"]).cpu().numpy()
            tgt_overlap_matrix = (
                extract_triu_without_diag(input_["overlap_matrix"]).cpu().numpy()
            )
            tgt_count_matrix = (
                extract_triu_without_diag(input_["count_matrix"]).cpu().numpy()
            )
            pred_depth = extract_triu_without_diag(output["depth"]).cpu().numpy()
            score_matrix = 2 / tgt_count_matrix
            self._n_instances.append(input_["depth_matrix"].shape[0])

            # from here we dont have any diag anymore
            mask_overlaps = defaultdict(list)
            mask_overlaps["ovlX"] = tgt_overlap_matrix == 0
            mask_overlaps["ovlO"] = tgt_overlap_matrix == 1
            mask_overlaps["ovlOX"] = mask_overlaps["ovlX"] | mask_overlaps["ovlO"]

            mask_eqs = defaultdict(list)
            mask_eqs["eq"] = tgt_depth == 2
            mask_eqs["neq"] = (tgt_depth == 0) | (tgt_depth == 1)
            mask_eqs["all"] = mask_eqs["eq"] | mask_eqs["neq"]

            whdr_per_overlap = defaultdict(list)
            for mask_ovl in mask_overlaps.keys():
                for mask_eq in mask_eqs.keys():
                    mask = mask_overlaps[mask_ovl] & mask_eqs[mask_eq]
                    whdr = self.calculate_whdr(
                        pred_depth, tgt_depth, score_matrix, mask
                    )
                    k = f"{mask_ovl}_{mask_eq}"
                    whdr_per_overlap[k].append(whdr)
            if self.sample_verbose:
                logger.info(
                    f"pred=\n{output['depth'].fill_diagonal_(-1)}\ntgt=\n{input_['depth_matrix'].fill_diagonal_(-1)}\nWHDR ovlX_all = {whdr_per_overlap['ovlX_all'][0]:.3f} ; WHDR ovlO_all = {whdr_per_overlap['ovlO_all'][0]:.3f} ; WHDR ovlOX_all = {whdr_per_overlap['ovlOX_all'][0]:.3f}"
                    # f"[Image : {input_['image_id']}] WHDR ovlX_all = {whdr_per_overlap['ovlX_all'][0]:.3f} ; WHDR ovlO_all = {whdr_per_overlap['ovlO_all'][0]:.3f} ; WHDR ovlOX_all = {whdr_per_overlap['ovlOX_all'][0]:.3f}"
                )

            """
            if (
                input_["depth_matrix"].shape[0] < 6
                and whdr_per_overlap["ovlOX_all"][-1] < 0.1
            ):
                file_name = input_["file_name"]
                self._fn.append(file_name)
            """

            for ovl_eq_str in whdr_per_overlap.keys():
                self._whdr[ovl_eq_str].append(whdr_per_overlap[ovl_eq_str][0])

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        def merge_dols(dol1: Dict[str, List[float]], dol2: Dict[str, List[float]]):
            """Merges dictionary of lists"""
            result = dict(dol1, **dol2)
            result.update((k, dol1[k] + dol2[k]) for k in set(dol1).intersection(dol2))
            return result

        comm.synchronize()
        self._whdr = comm.gather(self._whdr)
        self._n_instances = comm.gather(self._n_instances)

        # self._fn = comm.gather(self._fn)
        # self._fn = np.array(list(chain(*self._fn)))
        # np.savetxt("depth_res.txt", self._fn, fmt="%s")

        if not comm.is_main_process():
            return

        whdr = {}
        for proc in self._whdr:
            whdr = merge_dols(whdr, proc)
        self._whdr = whdr

        results = {}
        for ovl_eq_str in self._whdr.keys():  # each WHDR type
            whdr_arr = np.array(self._whdr[ovl_eq_str])
            mask_valid = whdr_arr != -1
            mean = whdr_arr[mask_valid].sum() / (len(whdr_arr[mask_valid]) + 1e-6)
            ovl_str, eq_str = ovl_eq_str.split("_")
            results[f"{ovl_str}/WHDR_{eq_str}"] = mean
            if self.multi_scale_evaluation:
                for i in range(min(self._n_instances), max(self._n_instances) + 1):
                    multi_scale_whdr = whdr_arr[self._n_instances == i]
                    mask_valid = multi_scale_whdr != -1
                    mean = multi_scale_whdr[mask_valid].sum() / (
                        len(multi_scale_whdr[mask_valid]) + 1e-6
                    )
                    ovl_str, eq_str = ovl_eq_str.split("_")
                    results[f"{ovl_str}/WHDR_{eq_str}/dim_{i}"] = mean

        ret = {"depth_order": results}
        # print_csv_format(ret)
        return ret
