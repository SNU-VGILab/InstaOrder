import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
INSTAORDER_PAN_NAME = "InstaOrderPanoptic_"
COCO_DIR = os.path.join(_root, "coco")
ANNS_DIR = os.path.join(COCO_DIR, "annotations")


# From detectron2.data.datasets.coco_panoptic source code.
# Modification : also adds InstaOrder annotations to the panoptic samples.
def load_instaorder_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(
            image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg"
        )
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        instaorder = ann["instaorder"]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "instaorder": instaorder,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def get_instaorder_panoptic(split: str):
    json_file = os.path.join(ANNS_DIR, f"{INSTAORDER_PAN_NAME}{split}2017.json")
    image_dir = os.path.join(COCO_DIR, f"{split}2017")
    gt_dir = os.path.join(COCO_DIR, f"panoptic_{split}2017")
    meta = MetadataCatalog.get(f"coco_2017_{split}_panoptic").as_dict()

    return load_instaorder_panoptic_json(json_file, image_dir, gt_dir, meta)


def get_instaorder_panoptic_train():
    dataset = get_instaorder_panoptic("train")
    return dataset


def get_instaorder_panoptic_val():
    dataset = get_instaorder_panoptic("val")
    return dataset


def set_instaorder_panoptic_metadata() -> None:
    """
    Adds metadata for InstaOrder Panoptic in the global metadata dictionary.
    """

    for split in ["train", "val"]:
        instaorder_metadata = MetadataCatalog.get(f"instaorder_2017_{split}_panoptic")

        coco_panoptic_metadata = MetadataCatalog.get(
            f"coco_2017_{split}_panoptic"
        ).as_dict()

        instaorder_panoptic_json_fn = f"{INSTAORDER_PAN_NAME}{split}2017.json"
        coco_panoptic_metadata.pop("name")

        coco_panoptic_metadata["evaluator_type"] = "instaorder"
        coco_panoptic_metadata["json_file"] = os.path.join(
            ANNS_DIR, instaorder_panoptic_json_fn
        )
        coco_panoptic_metadata["panoptic_json"] = os.path.join(
            ANNS_DIR, instaorder_panoptic_json_fn
        )
        instaorder_metadata.set(**coco_panoptic_metadata)


def register_instaorder_panoptic() -> None:
    """
    Registers InstaOrderPanoptic data and metadata for detectron2.
    We use the same convention as the other datasets.
    More precisely, the datasets can be summoned by calling
    `instaorder_2017_train_panoptic` and `instaorder_2017_val_panoptic`.
    """

    DatasetCatalog.register(
        "instaorder_2017_train_panoptic", get_instaorder_panoptic_train
    )
    DatasetCatalog.register("instaorder_2017_val_panoptic", get_instaorder_panoptic_val)
    set_instaorder_panoptic_metadata()


register_instaorder_panoptic()  # Register datasets on calling this file
