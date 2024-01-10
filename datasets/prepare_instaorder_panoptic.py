import os
import json
import multiprocessing as mp
from typing import List, Tuple, Sequence, Union
import argparse
import logging
import cvbase as cvb
from tqdm import tqdm
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from panopticapi.utils import rgb2id


logger = logging.getLogger("INSTAORDER_PANOPTIC_DATASET")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(name)s:%(levelname)s:%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

COCO_DIR = os.path.join(os.environ["DETECTRON2_DATASETS"], "coco")
COCO_ANNS = os.path.join(
    COCO_DIR,
    "annotations",
)
COCO_TRAIN = os.path.join(COCO_ANNS, "instances_train2017.json")
COCO_VAL = os.path.join(COCO_ANNS, "instances_val2017.json")
PANOPTIC_TRAIN = os.path.join(COCO_ANNS, "panoptic_train2017.json")
PANOPTIC_VAL = os.path.join(COCO_ANNS, "panoptic_val2017.json")
INSTAORDER_TRAIN = os.path.join(COCO_ANNS, "InstaOrder_train2017.json")
INSTAORDER_VAL = os.path.join(COCO_ANNS, "InstaOrder_val2017.json")

PAN_VAL_NB_SAMPLES = 5000  # Base panoptic dataset number of samples.


def read_coco(split):
    """
    Reads a coco-like dataset. Returns either a dictionary
    or a COCO instance if the dataset loaded is the original COCO

    Available keys :
    'panoptic_train', 'panoptic_val',
    'coco_train', 'coco_val',
    'instaorder_train', 'instaorder_val'
    """

    train_val_condition = "train" if "train" in split else "val"
    datasets = {
        f"panoptic_{train_val_condition}": [
            cvb.load,
            PANOPTIC_TRAIN if "train" in split else PANOPTIC_VAL,
        ],
        f"coco_{train_val_condition}": [
            COCO,
            COCO_TRAIN if "train" in split else COCO_VAL,
        ],
        f"instaorder_{train_val_condition}": [
            cvb.load,
            INSTAORDER_TRAIN if "train" in split else INSTAORDER_VAL,
        ],
    }
    dataset_loader = datasets[split]
    return dataset_loader[0](dataset_loader[1])


def read_all_data(split: str):
    """
    Returns all data from COCO, COCO panoptic and InstaOrder for a givne split, in that order.
    """
    assert split in [
        "train",
        "val",
    ], "Dataset split must be either of `train` or `val`."
    return (
        read_coco(f"coco_{split}"),
        read_coco(f"panoptic_{split}"),
        read_coco(f"instaorder_{split}"),
    )


def read_img(file_path: str) -> np.ndarray:
    return np.asarray(Image.open(file_path))


def get_masks_from_ann_ids(coco_obj: COCO, ann_ids: List[int]) -> np.ndarray:
    """
    Returns an numpy array of all the binary masks for the given annotation IDs.
    """
    anns = coco_obj.loadAnns(ann_ids)
    return np.array([coco_obj.annToMask(ann) for ann in anns])


def bin_mask_iou(mask1, mask2):
    """
    Computes the Intersection over Union (IoU) of two binary masks.
    """
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    union = mask1_area + mask2_area - intersection
    if union == 0:  # allows to compute IoUs between all 0 masks
        return 0
    else:
        return intersection / union


def is_binary_mask(array: np.ndarray) -> bool:
    return np.array_equal(array, array.astype(bool))


def binarize_mask_0_1(mask: np.ndarray) -> np.ndarray:
    """
    Given a binary mask, sets all its entries to 0 and 1.
    """
    assert is_binary_mask(
        mask
    ), "Mask must by binary to convert it to a {0, 1} binary mask."
    return np.clip(mask, 0, 1)


def subtract_mask(minuend: np.ndarray, subtrahend: np.ndarray) -> np.ndarray:
    """
    Computes the difference between two masks as the subtraction of their intersection.
    More specifically : m1 - m2 = m1 - inter(m1, m2)
    With m1 and m2 binary masks and inter the function that returns the intersection indices of the two masks.
    """
    assert is_binary_mask(minuend) and is_binary_mask(
        subtrahend
    ), "Minuend and subtrahend must be binary arrays to compute a difference between masks."
    minuend = binarize_mask_0_1(minuend)
    subtrahend = binarize_mask_0_1(subtrahend)
    minuend[np.where(minuend == subtrahend)] = 0
    return minuend


def reindex_dataset_on_img_ids(dataset: dict[str, List]) -> dict[int, dict]:
    """
    Reindexes all the annotations of a givend dataset on its 'image_id' field.
    """
    return {ann["image_id"]: ann for ann in dataset["annotations"]}


def create_empty_instaorder_panoptic(
    panoptic_template: dict[str, List]
) -> dict[str, Sequence]:
    """
    Creates an empty InstaOrderPanoptic dataset that contains all the keys of the base COCO panoptic dataset.
    The values for the fields 'annotations' and 'images' are initialized as empty lists.
    """
    instaorder_panoptic = {
        k: v
        for k, v in panoptic_template.items()
        if k != "annotations" and k != "images"
    }
    instaorder_panoptic["annotations"] = []
    instaorder_panoptic["images"] = []
    return instaorder_panoptic


def load_pan_gt(pan_file_name: str, split: str) -> np.ndarray:
    """Returns the panoptic GT from the associated .png file."""
    pan_gt_path = os.path.join(COCO_DIR, f"panoptic_{split}2017", pan_file_name)
    return rgb2id(read_img(pan_gt_path))


def determine_split(panoptic_len: int) -> str:
    """Determines the split of the given set based on the number of samples."""
    return "train" if panoptic_len > PAN_VAL_NB_SAMPLES else "val"


def compute_iou_between_all_pairs(
    instaorder_masks: np.ndarray, pan_masks: np.ndarray
) -> np.ndarray:
    """
    Returns a [pan, instances] shaped matrix where the (i, j)-th entry corresponds to the IoU
    between the i-th panoptic segment and the j-th instaorder segment.
    """
    return np.array(
        [
            list(
                map(
                    lambda instance_maks: bin_mask_iou(pan_mask, instance_maks),
                    instaorder_masks,
                )
            )
            for pan_mask in pan_masks
        ]
    )


def match_masks(
    instaorder_masks: np.ndarray, pan_masks: np.ndarray
) -> dict[Tuple[int, int], float]:
    """
    Iterative policy for matching segments IDs in InstaOrder (COCO instances masks) and COCO panoptic.
    This method solely relies on IoU.
    Iteratively computes IoU between unmatched segments and classifies high confidence segments in priority.
    Stopping conditions : either all InstaOrder instance IDs have been matched or there is no IoU score above 0.5 left to process.
    """
    matching = {}
    matched_masks = 0
    ious = compute_iou_between_all_pairs(instaorder_masks, pan_masks)
    while matched_masks < len(instaorder_masks) and ious.max() > 0.5:
        pan_idx, instaorder_idx = np.unravel_index(ious.argmax(), ious.shape)
        max_confidence = ious.max()
        matching[pan_idx, instaorder_idx] = max_confidence
        # Masking the indexes for the selected mask
        ious[pan_idx] = -1
        ious[..., instaorder_idx] = -1
        # Allowing to see which other masks are to be matched
        pan_unmatched_idx, instaorder_unmatched_idx = np.where(np.rint(ious) != -1)
        # Subtract the deleted mask on all of the other masks
        instaorder_masks = np.array(
            [
                subtract_mask(mask, instaorder_masks[instaorder_idx])
                for mask in instaorder_masks
            ]
        )
        pan_masks = np.array(
            [subtract_mask(mask, pan_masks[pan_idx]) for mask in pan_masks]
        )
        # Which allows to recompute the IoU between all the other masks.
        refined_ious = compute_iou_between_all_pairs(instaorder_masks, pan_masks)
        unmatched_masks = (pan_unmatched_idx, instaorder_unmatched_idx)
        ious[unmatched_masks] = refined_ious[unmatched_masks]

        matched_masks += 1
    return matching


def get_unmatched_instaorder_masks(
    matching: dict[Tuple[int, int], float], instaorder_ids: List[int]
) -> List[Tuple[int, int]]:
    """
    Given a matching dictionnary with keys (pan_idx, instance_idx) and instaorder_ids,
    all of the masks indexes and IDs that have not been matched.
    """
    instaorder_matched_idx = [m[1] for m in matching]
    masks_idx = [m for m, _ in enumerate(instaorder_ids)]
    unmatched_instaorder_idx = [m for m in masks_idx if m not in instaorder_matched_idx]
    unmatched_instaorder_ids = [
        instaorder_ids[unmatched_idx] for unmatched_idx in unmatched_instaorder_idx
    ]
    unmatched_instaorder_masks = [
        (idx, id) for idx, id in zip(unmatched_instaorder_idx, unmatched_instaorder_ids)
    ]
    return unmatched_instaorder_masks


def log_unmatched_masks(
    unmatched_masks: dict, unmatched_instaorder_masks: list, img_id: int
) -> dict[int, Tuple[int, int]]:
    has_unmatched_masks = len(unmatched_masks) > 0
    if has_unmatched_masks:
        logger.info(
            f"Found unmatched {'mask' if len(unmatched_instaorder_masks) == 1 else 'masks'} for image ID {img_id} : {unmatched_instaorder_masks}"
        )
        unmatched_masks[img_id] = unmatched_instaorder_masks
    return unmatched_masks


def add_images(
    instaorder_panoptic: dict[str, Sequence], panoptic: dict[str, Sequence]
) -> dict[str, Sequence]:
    """
    Adds the image field for InstaOrder Panoptic to make it consistent with the
    COCO Panoptic dataset fomat.
    """
    images_ids = list(
        map(lambda sample: sample["image_id"], instaorder_panoptic["annotations"])
    )
    images = [sample for sample in panoptic["images"] if sample["id"] in images_ids]
    instaorder_panoptic["images"] = images

    # dict is passed by reference hence return useless but maybe more clear
    return instaorder_panoptic


def generate_instaorder_pan_sample(
    matching: dict[Tuple[int, int], float],
    instaorder_sample: dict[str, Sequence],
    pan_sample: dict[str, Sequence],
) -> Union[None, dict[str, Sequence]]:
    """
    Generates the InstaOrderPanoptic sample given an InstaOrder and its corresponding COCO panoptic sample.
    If no geometric annotations are available, returns None.
    """
    has_geometric_relation = len(matching) > 1
    if has_geometric_relation:
        # Generate segments mapping.
        mapping = [-1 for _ in instaorder_sample["instance_ids"]]
        pan_ids = [ann["id"] for ann in pan_sample["segments_info"]]
        for pan_id, instance_id in matching:
            mapping[instance_id] = pan_ids[pan_id]
        instaorder_sample["pan_instance_ids"] = mapping

        instaorder_pan_sample = pan_sample
        instaorder_pan_sample["instaorder"] = instaorder_sample
        instaorder_pan_sample["instaorder"].pop("image_id")
        return instaorder_pan_sample
    return None


def merge_anns(
    coco_obj: COCO,
    panoptic: dict[str, dict],
    instaorder: dict[str, dict],
    empty_instaorder_panoptic: dict[str, Sequence],
) -> dict[str, Sequence]:
    """
    Given an InstaOrder, COCO instances and COCO panoptic dataset,
    merges the 3 to obtain the InstaOrderPanoptic dataset.
    """
    split = determine_split(len(panoptic["annotations"]))
    panoptic_reindex = reindex_dataset_on_img_ids(panoptic)
    instaorder_reindex = reindex_dataset_on_img_ids(instaorder)
    unmatched_masks = {}
    ann_count = []
    for pan_sample in tqdm(panoptic_reindex.values()):
        img_id = pan_sample["image_id"]
        instaorder_sample = instaorder_reindex.get(img_id)
        if instaorder_sample is not None:
            instaorder_ids = instaorder_sample["instance_ids"]
            pan_ids = [seg_info["id"] for seg_info in pan_sample["segments_info"]]
            instaorder_masks = get_masks_from_ann_ids(coco_obj, instaorder_ids)
            pan_gt = load_pan_gt(pan_sample["file_name"], split)
            pan_masks = np.array([pan_gt == id for id in pan_ids])
            # counter for masks, while they are not all tagged or if no match is above 50%.
            matching = match_masks(instaorder_masks, pan_masks)
            unmatched_instaorder_masks = get_unmatched_instaorder_masks(
                matching, instaorder_ids
            )
            unmatched_masks = log_unmatched_masks(
                unmatched_masks, unmatched_instaorder_masks, img_id
            )
            instaorder_pan_sample = generate_instaorder_pan_sample(
                matching, instaorder_sample, pan_sample
            )
            if instaorder_pan_sample is not None:
                empty_instaorder_panoptic["annotations"].append(instaorder_pan_sample)
            else:
                logger.info(
                    f"Dropping sample for image ID : {img_id}. No valid geometric relation annotations."
                )
            ann_count.append(True if instaorder_pan_sample is not None else False)

    instaorder_panoptic = add_images(empty_instaorder_panoptic, panoptic)
    logger.info(
        f"Found {len([len(unmatches) for unmatches in unmatched_masks.values()])} unmatched masks in {len(unmatched_masks)} images."
    )
    logger.info(f"Total of {len(ann_count)} samples processed.")
    logger.info(f"{ann_count.count(True)} valid samples added.")
    logger.info(f"{len(ann_count) - ann_count.count(True)} invalid samples excluded.")
    return instaorder_panoptic


def save_json(data: dict, name: str) -> None:
    logger.info(f"Saving {name}...")
    with open(name, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved {name}.")


def create_instaorder_panoptic(splits: list[str], dest_path: str) -> None:
    for split in splits:
        coco, panoptic, instaorder = read_all_data(split)
        instaorder_pan = create_empty_instaorder_panoptic(panoptic)
        instaorder_pan = merge_anns(
            coco_obj=coco,
            panoptic=panoptic,
            instaorder=instaorder,
            empty_instaorder_panoptic=instaorder_pan,
        )
        save_json(instaorder_pan, dest_path)


# def split_dataset(dataset: Sequence, num_splits: int):
#     k, m = divmod(len(dataset), num_splits)
#     return (
#         dataset[split * k + min(split, m) : (split + 1) * k + min(split + 1, m)]
#         for split in range(num_splits)
#     )


# def multi_core_create_instaorder_panoptic(panoptic):
#     """
#     Create dataset using multiple cores for faster merging.
#     """
#     # TODO: finish to implement this method
#     num_cores = mp.cpu_count()
#     dataset_splits = split_dataset(panoptic["annotations"], num_cores)


def main():
    parser = argparse.ArgumentParser(
        description="Generating InstaOrder Panoptic Dataset."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        dest="train",
        help="If specified, creates the training set for InstaOrder Panoptic.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        default=False,
        dest="val",
        help="If specified, creates the validation set for InstaOrder Panoptic.",
    )
    parser.add_argument(
        "--dest_path",
        action="store",
        default=COCO_ANNS,
        dest="dest_path",
        help="The location to which the generated json files will be stored. \
                Default to '$DETECTRON2_DATASETS/coco/annotations/'.",
    )
    args = parser.parse_args()

    splits = ["train" if args.train else None, "val" if args.val else None]
    splits = [split for split in splits if split is not None]

    create_instaorder_panoptic(splits, args.dest_path)


if __name__ == "__main__":
    main()
