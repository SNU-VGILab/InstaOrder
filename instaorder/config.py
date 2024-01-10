from detectron2.config import CfgNode as CN


def add_instaordernet_config(cfg):
    """
    Adds config for InstaOrderNet.
    """
    cfg.INPUT.AUGMENT = False

    # Switching to panoptic setup for M2F inference.
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

    # InstaOrder Mapper.
    cfg.INPUT.DATASET_MAPPER_NAME = "instaorder_panoptic"
    cfg.INPUT.RM_BIDIREC = 0  # occlusion
    cfg.INPUT.RM_OVERLAP = 0  # depth
    cfg.INPUT.OCC_SAMPLES_PER_CLASS = [2353545, 388323]  # 0.86 / 0.14
    cfg.INPUT.DEPTH_SAMPLES_PER_CLASS = [1345145, 1345145, 49398]  # {0, 1, 2}

    # Freezing modules options. NOTE: Modify in .yaml !
    cfg.MODEL.BACKBONE.FREEZE = False
    cfg.MODEL.SEM_SEG_HEAD.FREEZE = False
    cfg.MODEL.REMOVE_LOSSES = []

    cfg.MODEL.GEOMETRIC_PREDICTOR = CN()

    # Modify in .yaml
    cfg.MODEL.GEOMETRIC_PREDICTOR.NAME = ""
    cfg.MODEL.GEOMETRIC_PREDICTOR.TASK = ""  # {o, d, od}
    cfg.MODEL.GEOMETRIC_PREDICTOR.PAIRWISE = False  # parallel or pairwise
    cfg.MODEL.GEOMETRIC_PREDICTOR.INPUT_SIZE = 256  # pairwise
    cfg.MODEL.GEOMETRIC_PREDICTOR.USE_INSTAORDERNET_WEIGHTS = True  # pairwise
    cfg.MODEL.GEOMETRIC_PREDICTOR.INSTAORDERNET_WEIGHTS_PATH = (
        "/home/pierre/data/out/InstaOrder_ckpt/"
    )

    cfg.MODEL.GEOMETRIC_PREDICTOR.OPERATE_ON_MASK_EMBED = True  # parallel

    cfg.MODEL.GEOMETRIC_PREDICTOR.OCCLUSION_ON = False
    cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_ON = False

    cfg.MODEL.GEOMETRIC_PREDICTOR.OCCLUSION_WEIGHT = 5.0
    cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_WEIGHT = 5.0  # TODO: fill in
    cfg.MODEL.GEOMETRIC_PREDICTOR.DEPTH_LOSS = "balanced_softmax"
    # Modify here
    cfg.MODEL.GEOMETRIC_PREDICTOR.USE_GT_MASKS = False

    # Parallel geometric occlusion
    cfg.MODEL.GEOMETRIC_PREDICTOR.USE_ADAPTERS = True  # Better, but reduces PQ
    cfg.MODEL.GEOMETRIC_PREDICTOR.ADAPTER_DIM = 64
    cfg.MODEL.GEOMETRIC_PREDICTOR.ADAPTER_SCALE = 0.1

    cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_LAYERS = 8
    cfg.MODEL.GEOMETRIC_PREDICTOR.D_MODEL = 512
    cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_HEADS = 8
    cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_NUM_LAYERS = 2
    cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_EXPANSION = 4
    cfg.MODEL.GEOMETRIC_PREDICTOR.FILTER_TRIGGERED_QUERIES = True  # better
    cfg.MODEL.GEOMETRIC_PREDICTOR.METHOD = "stack"  # {"cat", "sep", "stack"}  # stack
    cfg.MODEL.GEOMETRIC_PREDICTOR.FUSE = False  # usually worse
    cfg.MODEL.GEOMETRIC_PREDICTOR.TRANSPOSE_QUERIES = False  # usually worse
    cfg.MODEL.GEOMETRIC_PREDICTOR.ATTN_BEFORE_POOL = True  # better
    cfg.MODEL.GEOMETRIC_PREDICTOR.POS_ENC = False  # worse
    cfg.MODEL.GEOMETRIC_PREDICTOR.LEARNABLE_PE = False  # worse
    cfg.MODEL.GEOMETRIC_PREDICTOR.MASK_NON_MASK_VALUES = True  # better
    cfg.MODEL.GEOMETRIC_PREDICTOR.ADD_INPUT_REPRESENTATIONS = True  # better
    # Single transformer
    cfg.MODEL.GEOMETRIC_PREDICTOR.QUERIES_INPUTS = False

    # sklearn recall, precision and F1 score zero_division param.
    cfg.SOLVER.OCCLUSION_ZERO_DIV = 1

    cfg.TEST.OCCLUSION_EVALUATION = False
    cfg.TEST.DEPTH_EVALUATION = False
    cfg.TEST.SAMPLE_VERBOSE = False
    cfg.TEST.MULTI_SCALE_EVALUATION = False

    cfg.WANDB = CN()
    cfg.WANDB.METRICS = False  # Settings to false disables all WandB features.
    cfg.WANDB.VISUALIZATION = False
    if not cfg.WANDB.METRICS:
        # Cannot visualize if wandb log is set to false...
        cfg.WANDB.VISUALIZATION = False
    else:
        try:
            import wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb settings turned on even though package is not installed. Please install wandb."
            )
