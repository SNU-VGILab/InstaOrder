# Copyright (c) Facebook, Inc. and its affiliates.panoptic_occ
# Modified by Bowen Cheng from
# https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
from typing import Tuple, Union, List, Sequence, Dict

import numpy as np

import cv2

import torch
import torchvision.transforms as transforms

from pycocotools.coco import COCO
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.structures import BitMasks, Boxes, Instances


__all__ = ["MaskFormerPanopticOcclusionDatasetMapper"]


def build_transform_gen(cfg, is_train: bool) -> List[T.Transform]:
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """

    # train
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE
    # test
    max_size_test = cfg.INPUT.MAX_SIZE_TEST
    min_size_test = cfg.INPUT.MIN_SIZE_TEST

    augmentation = []

    if is_train:
        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        # augmentation.extend(
        #     [
        #         T.ResizeScale(
        #             min_scale=min_scale,
        #             max_scale=max_scale,
        #             target_height=image_size,
        #             target_width=image_size,
        #         ),
        #         T.FixedSizeCrop(crop_size=(image_size, image_size)),
        #     ]
        # )
    else:
        augmentation.append(
            T.ResizeShortestEdge(
                short_edge_length=(min_size_test, min_size_test),
                max_size=max_size_test,
                sample_style="choice",
            )
        )
    return augmentation


class MaskFormerPanopticOcclusionDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.
    This dataset mapper applies the same transformation
    as DETR for COCO panoptic segmentation.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        tfm_gens,
        image_format,
        geometric_task: str,
        occ_input_size: int,
        rm_bidirec: int,
        rm_overlap: int,
        use_gt_masks: bool,
        augment: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic
                           transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by
                          :func:`detection_utils.read_image`.
        """

        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            f"Full TransformGens used in {'training' if is_train else 'testing'}: {str(self.tfm_gens)}"
        )

        self.img_format = image_format
        self.is_train = is_train

        # InstaOrder
        self.geometric_task = geometric_task
        self.occ_input_size = occ_input_size
        self.rm_bidirec = rm_bidirec
        self.rm_overlap = rm_overlap

        self.use_gt_masks = use_gt_masks
        if use_gt_masks:
            coco_type = "train2017" if is_train else "val2017"
            self.coco = COCO(
                annotation_file=f"/home/pierre/data/coco/annotations/instances_{coco_type}.json",
            )

        # NOTE: debug only
        self.augment = augment

    @classmethod
    def from_config(cls, cfg, is_train=True):
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "geometric_task": cfg.MODEL.GEOMETRIC_PREDICTOR.TASK,
            "rm_bidirec": cfg.INPUT.RM_BIDIREC,
            "rm_overlap": cfg.INPUT.RM_OVERLAP,
            "use_gt_masks": cfg.MODEL.GEOMETRIC_PREDICTOR.USE_GT_MASKS,
            "occ_input_size": cfg.MODEL.GEOMETRIC_PREDICTOR.INPUT_SIZE,
            "augment": cfg.INPUT.AUGMENT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image,
                                 in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        # First line will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )  # using PIL RGB
        dataset_dict["low_image"] = prepare_for_instaorder_net(
            image, self.occ_input_size
        )

        utils.check_image_size(dataset_dict, image)

        if self.augment:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor
        # due to shared-memory, but not efficient on large generic
        # data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            if self.augment:
                pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                # NOTE: Since the "iscrowd" annotations are not considered as instances,
                # InstaOrderPanoptic indexes the segments discarding 'iscrowd' annotations.
                # So the indexing from InstaOrderPanoptic GT is aligned with the 'instances'
                # field.
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros(
                    (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
                )
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack(
                        [
                            torch.from_numpy(np.ascontiguousarray(x.copy()))
                            for x in masks
                        ]
                    )
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

                dataset_dict["instances"] = instances

            # InstaOrder dataset
            instaorder = dataset_dict.pop("instaorder")
            occlusion = instaorder["occlusion"]
            depth = instaorder["depth"]
            pan_instance_ids = instaorder["pan_instance_ids"]
            (
                pan_seg_ids_annotated_instaorder,
                pan_seg_idx_annotated_instaorder,
            ) = get_pan_ids_idx_instaorder_annotated(segments_info, pan_instance_ids)
            gt_idx_to_pan_ids = get_gt_permutation_to_pan_seg_idx(
                pan_seg_ids_annotated_instaorder, pan_instance_ids
            )
            dataset_dict["tgt_pan_idx"] = torch.tensor(pan_seg_idx_annotated_instaorder)

            if self.geometric_task == "o" or self.geometric_task == "od":
                gt_occlusion_matrix = get_occlusion_matrix(
                    occlusion,
                    len(pan_instance_ids),
                    rm_bidirec=self.rm_bidirec,
                )
                if not self.use_gt_masks:
                    # pan segs idx annotated in InstaOrder
                    gt_occlusion_matrix = reindex_gt_matrix(
                        gt_occlusion_matrix, gt_idx_to_pan_ids
                    )
                dataset_dict["occlusion_matrix"] = gt_occlusion_matrix
            if self.geometric_task == "d" or self.geometric_task == "od":
                gt_depth_matrices = get_depth_overlap_count_matrices(
                    depth, len(pan_instance_ids), rm_overlap=self.rm_overlap
                )
                if not self.use_gt_masks:
                    # pan segs idx annotated in InstaOrder
                    gt_depth_matrices = [
                        reindex_gt_matrix(gt_matrix, gt_idx_to_pan_ids)
                        for gt_matrix in gt_depth_matrices
                    ]
                dataset_dict["depth_matrix"] = gt_depth_matrices[0]
                dataset_dict["overlap_matrix"] = gt_depth_matrices[1]
                dataset_dict["count_matrix"] = gt_depth_matrices[2]
                # print(f"{depth}\n{gt_depth_matrices[0]}\n{gt_depth_matrices[1]}")

            if self.use_gt_masks:
                dataset_dict["low_masks"] = get_coco_masks_for_instaorder_sample(
                    instaorder, self.coco, self.occ_input_size
                )
        return dataset_dict


def get_pan_ids_idx_instaorder_annotated(
    segments_info: List[Dict[str, Sequence]], annotated_pan_instance_ids: List[int]
) -> Tuple[List[int], List[int]]:
    """
    Returns the `isthings` and not `iscrowd` panoptic segments IDs and indices
    as two lists.
    The first one contains the IDs and the second the index at which they appear
    in the `segments_info` dictionary.
    """
    ids_and_idx = [
        (seg["id"], i)
        for i, seg in enumerate(segments_info)
        if seg["isthing"]
        and not seg["iscrowd"]
        and seg["id"] in annotated_pan_instance_ids
    ]
    return map(list, zip(*ids_and_idx))


def get_gt_permutation_to_pan_seg_idx(
    segments_ids: List[int],
    pan_instance_ids: List[int],
) -> torch.Tensor:
    occ_to_pan_seg_idx = [pan_instance_ids.index(id) for id in segments_ids]
    return torch.tensor(occ_to_pan_seg_idx)


def reindex_gt_matrix(
    gt_occlusion: torch.Tensor,
    occ_idx_to_pan_ids: torch.Tensor,
) -> torch.Tensor:
    gt_occlusion = gt_occlusion[occ_idx_to_pan_ids][:, occ_idx_to_pan_ids]
    return gt_occlusion


def get_occlusion_matrix(
    gt_occlusion: List[Dict[str, Union[str, int]]],
    nb_instances: int,
    rm_bidirec: int,
) -> torch.Tensor:
    """
    Gets the ground truth matrix of a given sample for a given type
    (occlusion or depth).

    Parameters
    ----------
    instaorder_info: `dict`
        A dictionary representing the current sample.
    rm_bidirec: `int`
        An integer `{0, 1}` specifying whether or
        not to keep the bidirection.

    Returns
    -------
    occ_matrix: `np.array`
        The ground truth occlusion matrix.
    """
    # Init occlusion matrix.
    gt_occ_matrix = torch.zeros((nb_instances, nb_instances), dtype=torch.int)

    if len(gt_occlusion) == 0:
        return gt_occ_matrix
    for relation in gt_occlusion:  # dict containing 'order' and 'count'.
        # TODO: fix this
        if "&" in relation["order"] and rm_bidirec == 1:
            gt_occ_matrix[instance1, instance2] = -1
            gt_occ_matrix[instance2, instance1] = -1

        elif "&" in relation["order"]:
            instance1, instance2 = list(
                map(int, relation["order"].split(" & ")[0].split("<"))
            )
            gt_occ_matrix[instance1, instance2] = 1
            gt_occ_matrix[instance2, instance1] = 1

        else:
            instance1, instance2 = list(map(int, relation["order"].split("<")))
            gt_occ_matrix[instance1, instance2] = 1
    return gt_occ_matrix  # nb_instances x nb_instances


def get_depth_overlap_count_matrices(
    gt_depth: List[Dict[str, str]], nb_instances: int, rm_overlap: int
):
    gt_depth_matrix = torch.zeros((nb_instances, nb_instances), dtype=torch.int)
    is_overlap_matrix = torch.zeros((nb_instances, nb_instances), dtype=torch.int)
    # There is at least one count...
    count_matrix = torch.ones((nb_instances, nb_instances), dtype=torch.int)
    # print(
    #     f"init gt_depth\n{gt_depth_matrix}\ninit ngt_overlap\n{is_overlap_matrix}\ninit count\n{count_matrix}"
    # )

    if len(gt_depth) == 0:
        return [gt_depth_matrix, is_overlap_matrix, count_matrix]

    for overlap_count in gt_depth:
        depth_order = overlap_count["order"]
        is_overlap = overlap_count["overlap"]
        count = overlap_count["count"]
        # print(f"depth order {depth_order}")
        # print(f"is_ovl {is_overlap}")
        # print(f"count {count}")

        split_char = "<" if "<" in depth_order else "="
        idx1, idx2 = list(map(int, depth_order.split(split_char)))
        if rm_overlap and is_overlap:
            is_overlap_matrix[idx1, idx2] = -1
            is_overlap_matrix[idx2, idx1] = -1

        # set is_overlap_matrix
        elif is_overlap:
            is_overlap_matrix[idx1, idx2] = 1
            is_overlap_matrix[idx2, idx1] = 1
        else:
            is_overlap_matrix[idx1, idx2] = 0
            is_overlap_matrix[idx2, idx1] = 0

        # set gt_depth_matrix
        if split_char == "<":
            gt_depth_matrix[idx1, idx2] = 1
            gt_depth_matrix[idx2, idx1] = 0
        elif split_char == "=":
            gt_depth_matrix[idx1, idx2] = 2
            gt_depth_matrix[idx2, idx1] = 2
        # set count_matrix
        count_matrix[idx1, idx2] = count
        count_matrix[idx2, idx1] = count
    # Following InstaOrder
    gt_depth_matrix.fill_diagonal_(-1)
    is_overlap_matrix.fill_diagonal_(-1)
    count_matrix.fill_diagonal_(-1)
    # print(
    #     f"gt_depth\n{gt_depth_matrix}\ngt_overlap\n{is_overlap_matrix}\ncount\n{count_matrix}"
    # )
    return [gt_depth_matrix, is_overlap_matrix, count_matrix]  # num x num


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample


def prepare_for_instaorder_net(image: np.ndarray, out_size: int) -> torch.Tensor:
    """Reproduces the `resize` config of InstaOrderNet"""
    instaorder_mean = [0.485, 0.456, 0.406]
    instaorder_std = [0.229, 0.224, 0.225]

    transform_low_img = transforms.Compose(
        [
            Resize(
                out_size,
                out_size,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(instaorder_mean, instaorder_std),
            PrepareForNet(),
        ]
    )
    image = transform_low_img({"image": image / 255.0})["image"].astype(np.float32)
    return torch.from_numpy(image)


def resize_masks(masks, out_size: int):
    import cv2

    low_res_masks = [
        cv2.resize(
            mask,
            (out_size, out_size),
            interpolation=cv2.INTER_NEAREST,
        )
        for mask in masks
    ]
    return low_res_masks


def get_coco_masks_for_instaorder_sample(
    instaorder_sample: Dict, coco: COCO, out_size: int
):
    anns_ids = instaorder_sample["instance_ids"]
    anns = coco.loadAnns(anns_ids)
    masks = [coco.annToMask(ann) for ann in anns]
    masks = np.array(resize_masks(masks, out_size), dtype=np.uint8)
    return torch.from_numpy(masks)
