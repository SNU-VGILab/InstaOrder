from typing import List, Tuple, Dict
from itertools import chain

import numpy as np

import torch
from detectron2.utils.logger import setup_logger
from detectron2.config import CfgNode, configurable
from detectron2.utils import comm
from detectron2.engine.defaults import DatasetEvaluator, print_csv_format

from sklearn.metrics import precision_score, recall_score, f1_score


logger = setup_logger(name="d2.evaluation.evaluator.occlusion")

try:
    import wandb
except ModuleNotFoundError:
    logger.info("Wandb not installed")


class OcclusionEvaluator(DatasetEvaluator):
    @configurable
    def __init__(
        self,
        occlusion_zero_div: int,
        sample_verbose: bool,
        wandb_visualization: bool,
        multi_scale_evaluation: bool = False,
    ):
        self.zero_div = occlusion_zero_div
        self.sample_verbose = sample_verbose
        self.wandb_visualization = wandb_visualization
        self.multi_scale_evaluation = multi_scale_evaluation

    @classmethod
    def from_config(cls, cfg: CfgNode):
        return {
            "occlusion_zero_div": cfg.SOLVER.OCCLUSION_ZERO_DIV,
            "sample_verbose": cfg.TEST.SAMPLE_VERBOSE,
            "wandb_visualization": cfg.WANDB.VISUALIZATION,
            "multi_scale_evaluation": cfg.TEST.MULTI_SCALE_EVALUATION,
        }

    def reset(self) -> None:
        self._recall = []
        self._precision = []
        self._f1 = []
        self._n_instances = []
        # self._fn = []

    def remove_diag_and_flatten(self, matrix: torch.Tensor) -> torch.Tensor:
        n_instances = matrix.shape[0]
        non_diag_idx = ~torch.eye(n_instances, dtype=torch.bool)
        filtered_matrix = matrix[non_diag_idx]
        return filtered_matrix.cpu().numpy()

    def eval_order_recall_precision_f1(
        self, order_matrix: torch.Tensor, gt_order_matrix: torch.Tensor
    ) -> Tuple:
        # print(
        #     f"pred :\n{order_matrix.fill_diagonal_(-1)}, tgt :\n{gt_order_matrix.fill_diagonal_(-1)}"
        # )
        order = self.remove_diag_and_flatten(order_matrix)
        gt_order = self.remove_diag_and_flatten(gt_order_matrix)
        recall = recall_score(
            gt_order, order, average="binary", zero_division=self.zero_div
        )
        precision = precision_score(
            gt_order, order, average="binary", zero_division=self.zero_div
        )
        f1 = f1_score(gt_order, order, average="binary", zero_division=self.zero_div)
        return recall * 100, precision * 100, f1 * 100

    def process(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        outputs: List[Dict[str, torch.Tensor]],
    ) -> None:
        for input_, output in zip(inputs, outputs):
            pred_occ = output["occlusion"]
            tgt_occ = input_["occlusion_matrix"]
            n_instances = tgt_occ.shape[0]
            recall, precision, f1 = self.eval_order_recall_precision_f1(
                pred_occ, tgt_occ
            )
            self._precision.append(precision)
            self._recall.append(recall)
            self._f1.append(f1)
            self._n_instances.append(n_instances)
            if self.sample_verbose:
                logger.info(
                    f"Recall : {recall:.3f} ; Precision {precision:.3f} ; F1 : {f1:.3f}"
                )

            """
            if f1 > 99.9 and n_instances < 6:
                file_name = input_["file_name"]
                self._fn.append(file_name)
            """

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        def mean(array: list) -> float:
            return sum(array) / len(array)

        comm.synchronize()
        self._precision = comm.gather(self._precision)
        self._recall = comm.gather(self._recall)
        self._f1 = comm.gather(self._f1)
        self._n_instances = comm.gather(self._n_instances)

        # self._fn = comm.gather(self._fn)

        self._precision = list(chain(*self._precision))
        self._recall = list(chain(*self._recall))
        self._f1 = list(chain(*self._f1))
        self._n_instances = list(chain(*self._n_instances))

        # self._fn = np.array(list(chain(*self._fn)))
        # np.savetxt("occ_res.txt", self._fn, fmt="%s")

        if not comm.is_main_process():
            return

        results = {}
        results["recall"] = mean(self._recall)
        results["precision"] = mean(self._precision)
        results["f1"] = mean(self._f1)
        ret = {"occlusion_order": results}

        if self.multi_scale_evaluation:
            self._recall = np.array(self._recall)
            self._precision = np.array(self._precision)
            self._f1 = np.array(self._f1)
            self._n_instances = np.array(self._n_instances)
            for i in range(min(self._n_instances), max(self._n_instances) + 1):
                results_dim = {}
                results_dim["recall"] = mean(self._recall[self._n_instances == i])
                results_dim["precision"] = mean(self._precision[self._n_instances == i])
                results_dim["f1"] = mean(self._f1[self._n_instances == i])
                ret[f"occlusion_order/dim_{i}"] = results_dim
        if self.wandb_visualization:
            wandb.log(results)
        # print_csv_format(ret)

        return ret
