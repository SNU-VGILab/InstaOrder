# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import pickle

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from instaorder import add_instaordernet_config
from predictor import VisualizationDemo
from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    _PanopticPrediction,
    _create_text_labels,
)


# constants
WINDOW_NAME = "mask2former demo"


class CustomVisualizationDemo(VisualizationDemo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = CustomVisualizer(
            img_rgb=image, metadata=self.metadata, instance_mode=self.instance_mode
        )
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


_OFF_WHITE = (1.0, 1.0, 240.0 / 255)


class CustomVisualizer(Visualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def draw_panoptic_seg(
        self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7
    ):
        """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata.stuff_classes[category_idx]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=None,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        labels = _create_text_labels(
            category_ids,
            scores,
            self.metadata.thing_classes,
            [x.get("iscrowd", 0) for x in sinfo],
        )

        try:
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in category_ids
            ]
        except AttributeError:
            colors = None
        self.overlay_instances(
            masks=masks, labels=None, assigned_colors=colors, alpha=alpha
        )

        return self.output

    draw_panoptic_seg_predictions = draw_panoptic_seg  # backward compatibility


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_instaordernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def save_geom_peds(predictions, save_dir, out_filename):
    out_filename = remove_file_extension(out_filename)
    pred_filename = os.path.join(save_dir, "prediction", out_filename)
    has_occ = True if predictions.get("occlusion") is not None else False
    has_depth = True if predictions.get("depth") is not None else False
    pred_type = "o" if has_occ else ""
    pred_type += "d" if has_depth else ""

    if has_occ:
        predictions["occlusion"] = predictions["occlusion"].cpu().numpy()
    if has_depth:
        predictions["depth"] = predictions["depth"].cpu().numpy()
    # predictions["panoptic_seg"][0] = predictions["panoptic_seg"][0].cpu().numpy()
    predictions["panoptic_seg"] = list(predictions["panoptic_seg"])
    predictions["panoptic_seg"][0] = predictions["panoptic_seg"][0].cpu().numpy()
    predictions["panoptic_seg"] = tuple(predictions["panoptic_seg"])

    with open(f"{pred_filename}_{pred_type}.pkl", "wb") as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
    """
    occlusion_r = predictions.get("occlusion")
    depth_r = predictions.get("depth")
    if occlusion_r is not None:
        print(occ_out_filename)
        np.savetxt(f"{occ_out_filename}.out", occlusion_r.cpu().numpy(), fmt="%d")
    if depth_r is not None:
        np.savetxt(f"{depth_out_filename}.out", depth_r.cpu().numpy(), fmt="%d")
    """


def remove_file_extension(file_path):
    dir, file_name = os.path.split(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name)
    path_without_extension = os.path.join(dir, file_name_without_extension)
    return path_without_extension


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # demo = VisualizationDemo(cfg)
    demo = CustomVisualizationDemo(cfg=cfg)

    if args.input:
        if len(args.input) == 1:
            if args.input[0].endswith("txt"):  # file containing inputs location
                with open(args.input[0], "r") as f:
                    args.input = f.read().splitlines()
            else:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    # TODO: make sure path exists or create folder/files.
                    seg_dir = os.path.join(args.output, "segmentation")
                    occ_dir = os.path.join(args.output, "occlusion")
                    depth_dir = os.path.join(args.output, "depth")
                    if not os.path.isdir(seg_dir):
                        os.makedirs(seg_dir)
                    if (
                        cfg.MODEL.GEOMETRIC_PREDICTOR.OCCLUSION_ON
                        and not os.path.isdir(occ_dir)
                    ):
                        os.makedirs(occ_dir)
                    if cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_ON and not os.path.isdir(
                        depth_dir
                    ):
                        os.makedirs(depth_dir)
                    out_filename = os.path.basename(path)
                    out_dir = args.output
                else:
                    assert (
                        len(args.input) == 1
                    ), "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(os.path.join(seg_dir, out_filename))
                if (
                    cfg.MODEL.GEOMETRIC_PREDICTOR.OCCLUSION_ON
                    or cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_ON
                ):
                    print(out_filename)
                    save_geom_peds(predictions, out_dir, out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv")
            if test_opencv_video_format("x264", ".mkv")
            else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
