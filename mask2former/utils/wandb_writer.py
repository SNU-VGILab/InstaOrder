import logging
from typing import Union, Dict

from detectron2.config import CfgNode
from detectron2.utils.events import EventWriter, get_event_storage


_VALID_TYPES = {tuple, list, str, int, float, bool}


class WandbWriter(EventWriter):
    """
    Logging metrics to WandB.
    """

    def __init__(
        self,
        project: str = "instaorder_panoptic",
        config: Union[Dict, CfgNode] = {},
        window_size: int = 20,
        **kwargs,
    ):
        """
        Args:
            project (str): W&B Project name
            config Union[Dict, CfgNode]: the project level configuration object
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `wandb.init(...)`
        """
        import wandb

        self._window_size = window_size
        self._run = (
            wandb.init(project=project, config=convert_to_dict(config), **kwargs)
            if not wandb.run
            else wandb.run
        )
        self._run._label(repo="mask2former")

    def write(self):
        storage = get_event_storage()

        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v

        self._run.log(log_dict)

    def close(self):
        self._run.finish()


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.getLogger(__name__).info(
                f"Key {'.'.join(key_list)} with value {type(cfg_node)}"
                " is not a WandB valid type."
                " It will not be logged on the WandB config writer..."
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict
