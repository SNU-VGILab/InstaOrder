from typing import List
from collections import defaultdict

import os

from .registry import GEOMETRIC_PREDICTOR_REGISTRY
from .resnet_cls import resnet50_cls
from .fixmodule import FixModule
from ..transformer_decoder.position_encoding import PositionEmbeddingSine

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from detectron2.config import configurable
from detectron2.utils.logger import setup_logger

logger = setup_logger(name="d2.modeling.geometrical_predictor")

__all__ = ["build_geometric_predictor"]


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@GEOMETRIC_PREDICTOR_REGISTRY.register()
class PairwiseResNet(nn.Module):
    @configurable
    def __init__(
        self,
        geometric_task: str,
        use_instaorder_weights: bool,
        instaordernet_weights_path: str,
    ):
        super().__init__()
        self.use_instaorder_weights = use_instaorder_weights
        self.geometric_task = geometric_task
        num_classes = {"o": 2, "d": 3, "od": [2, 3]}[geometric_task]
        instaordernet_cfg = {
            "in_channels": 5,
            "num_classes": num_classes,
        }
        self.instaordernet_weights_path = instaordernet_weights_path
        self.instaordernet = self.load_instaordernet_weights(**instaordernet_cfg)

    @classmethod
    def from_config(cls, cfg):
        return {
            "geometric_task": cfg.MODEL.GEOMETRIC_PREDICTOR.TASK,
            "use_instaorder_weights": cfg.MODEL.GEOMETRIC_PREDICTOR.USE_INSTAORDERNET_WEIGHTS,
            "instaordernet_weights_path": cfg.MODEL.GEOMETRIC_PREDICTOR.INSTAORDERNET_WEIGHTS_PATH,
        }

    def load_instaordernet_weights(self, **kwargs):
        instaordernet = resnet50_cls(**kwargs)

        if self.use_instaorder_weights:
            path = os.path.join(
                self.instaordernet_weights_path,
                f"InstaOrder_InstaOrderNet_{self.geometric_task}.pth.tar",
            )
            logger.info(f"Loading pre-trained InstaOrderNet weights from {path}")
            weights = torch.load(path)
            # Reproduce InstaOrderNet_o structure to make weights match
            instaordernet = FixModule(instaordernet)
            instaordernet.load_state_dict(weights["state_dict"])

        return instaordernet

    def forward(self, masks_and_images: torch.Tensor):
        return self.instaordernet(masks_and_images)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_num_layers: int,
        expansion: int = 4,
        cross_attn: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        self.ca = cross_attn
        if cross_attn:
            self.norm_ca = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * expansion, d_model, mlp_num_layers)

    def forward(self, queries, keys=None, values=None, attn_mask=None):
        norm = self.norm1(queries)
        attn = self.attn(norm, norm, norm)[0] + queries
        if self.ca:
            assert keys is not None and values is not None
            norm_attn = self.norm_ca(attn)
            if attn_mask is not None:
                attn = self.cross_attn(norm_attn, keys, values, attn_mask)[0] + attn
            else:
                attn = self.cross_attn(norm_attn, keys, values)[0] + attn
        norm = self.norm2(attn)
        logits = self.mlp(norm) + attn
        return logits


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        mlp_num_layers: int,
        method: str = "",
        expansion: int = 4,
        cross_attn: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        if method == "cat":
            d_model *= 2
        self.d_model = d_model
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    d_model, num_heads, mlp_num_layers, expansion, cross_attn
                )
            )

    def forward(
        self,
        x,
        cross_attn=None,
        attn_mask: torch.Tensor = None,
        return_intermediate: bool = True,
    ):
        preds = []
        for layer in self.layers:
            x = layer(x, cross_attn, cross_attn, attn_mask)
            preds.append(x)
        return preds if return_intermediate else preds[-1]


@GEOMETRIC_PREDICTOR_REGISTRY.register()
class GeometricTransformer(nn.Module):
    @configurable
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        geometric_tasks: str,
        mlp_num_layers: int,
        expansion: int,
        transpose_queries: bool,
        method: str,
        fuse: bool,
        attn_before_pool: bool,
        pos_enc: bool,
        learnable_pe: bool,
        mask_non_mask_values: bool,
        add_input_representations: bool,
        pooling_type=F.max_pool2d,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.task_names = []
        if "o" in geometric_tasks:
            self.task_names.append("pred_occlusion")
        if "d" in geometric_tasks:
            self.task_names.append("pred_depth")

        self.method = method
        self.fuse = fuse
        self.num_heads = num_heads
        self.mask_non_mask_values = mask_non_mask_values
        self.add_input_representations = add_input_representations

        self.pos_enc = pos_enc
        if self.pos_enc:
            self.pe = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=False)
        self.learnable_pe = learnable_pe
        if self.learnable_pe:
            self.pe = nn.Embedding(1, d_model)
        self.attn_before_pool = attn_before_pool
        if attn_before_pool:
            # TODO: find parameter in config that contains the dim output of m2f
            self.features_transformer = Transformer(
                256, num_heads, 1, mlp_num_layers, expansion=expansion
            )
        self.proj_feats_to_input_size = (
            nn.Linear(256, d_model) if d_model != 256 else None
        )
        self.proj_embed_to_input_size = (
            nn.Linear(256, d_model) if d_model != 256 else None
        )

        self.pooling = pooling_type

        # self.embed_mlp = nn.Linear(d_model, d_model, bias=False)
        # self.features_mlp = MLP(d_model, d_model * expansion, d_model, mlp_num_layers)
        if self.method == "sep":
            self.embed_transformer = Transformer(
                d_model,
                num_heads,
                num_layers,
                mlp_num_layers,
                method,
                expansion,
            )
            self.features_transformer = Transformer(
                d_model,
                num_heads,
                num_layers,
                mlp_num_layers,
                method,
                expansion,
            )
        if self.method in ["cat", "stack"]:
            self.concat_transformer = Transformer(
                d_model,
                num_heads,
                num_layers,
                mlp_num_layers,
                method,
                expansion,
            )
        self.projs = nn.ModuleList([])
        for _ in self.task_names:
            self.projs.append(
                nn.ModuleList(
                    MLP(d_model, d_model * expansion, d_model, mlp_num_layers)
                    for _ in range(2)  # one for each transformer
                )
            )
        # self.replace_mm = nn.Linear(256, 9 * 2)
        # self.norms = nn.ModuleList([])
        # for _ in self.task_names:
        #     self.norms.append(nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)]))

        # TODO: make head a bit more deep
        self.out_projs = nn.ModuleList([])
        for task in self.task_names:
            self.out_projs.append(
                MLP(
                    1 if not fuse else 2,
                    d_model,
                    2 if "occlusion" in task else 3,
                    mlp_num_layers,
                )
            )

        self.transpose_queries = transpose_queries
        self.d_model = d_model

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.GEOMETRIC_PREDICTOR.D_MODEL,
            "num_heads": cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_HEADS,
            "num_layers": cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_LAYERS,
            "geometric_tasks": cfg.MODEL.GEOMETRIC_PREDICTOR.TASK,
            "mlp_num_layers": cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_NUM_LAYERS,
            "expansion": cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_EXPANSION,
            "transpose_queries": cfg.MODEL.GEOMETRIC_PREDICTOR.TRANSPOSE_QUERIES,
            "method": cfg.MODEL.GEOMETRIC_PREDICTOR.METHOD,
            "fuse": cfg.MODEL.GEOMETRIC_PREDICTOR.FUSE,
            "attn_before_pool": cfg.MODEL.GEOMETRIC_PREDICTOR.ATTN_BEFORE_POOL,
            "pos_enc": cfg.MODEL.GEOMETRIC_PREDICTOR.POS_ENC,
            "learnable_pe": cfg.MODEL.GEOMETRIC_PREDICTOR.LEARNABLE_PE,
            "mask_non_mask_values": cfg.MODEL.GEOMETRIC_PREDICTOR.MASK_NON_MASK_VALUES,
            "add_input_representations": cfg.MODEL.GEOMETRIC_PREDICTOR.ADD_INPUT_REPRESENTATIONS,
        }

    def init_preds_dict(self):
        preds = defaultdict(list)
        for task in self.task_names:
            preds[task] = [[] for _ in range(self.num_layers)]
        return preds

    def add_to_preds(self, preds, layer_preds):
        for i, task in enumerate(self.task_names):
            for layer in range(self.num_layers):
                preds[task][layer].append(layer_preds[layer][i])
        return preds

    def forward_non_uniform(
        self,
        mask_embeddings: List[torch.Tensor],
        mask_features: List[torch.Tensor],
    ) -> defaultdict:
        preds = self.init_preds_dict()

        features = self.pool_features(mask_features)
        for mask_embed, mask_feature in zip(mask_embeddings, features):
            torch.cuda.empty_cache()
            # mask_embed = self.embed_mlp(mask_embed)
            # mask_features = self.features_mlp(mask_feature)
            # mask_embed = self.embed_transformer(mask_embed, return_intermediate=False)
            if self.proj_embed_to_input_size is not None:
                mask_embed = self.proj_embed_to_input_size(mask_embed)
                mask_feature = self.proj_feats_to_input_size(mask_feature)
            if self.method == "stack":
                embeddings = torch.cat((mask_embed, mask_feature), dim=0)
            if self.method == "cat":
                embeddings = torch.cat((mask_embed, mask_feature), dim=1)
            if self.method in ["cat", "stack"]:
                interaction = self.concat_transformer(embeddings)  # list of emb/feat
            if self.method == "stack":
                embeddings = [inter.chunk(2)[0] for inter in interaction]
                feats = [inter.chunk(2)[1] for inter in interaction]
            if self.method == "cat":
                embeddings = [inter[..., : self.d_model] for inter in interaction]
                feats = [inter[..., self.d_model :] for inter in interaction]
            if self.method == "sep":
                embeddings = self.embed_transformer(mask_embed)  # list of embeddings
                feats = self.features_transformer(mask_feature)  # list of features
            layer_preds = []
            for layer_embed, layer_feat in zip(embeddings, feats):
                task_per_layer = []
                for (embed_mlp, feat_mlp), out_proj in zip(self.projs, self.out_projs):
                    layer_embed = embed_mlp(layer_embed)
                    layer_feat = feat_mlp(layer_feat)
                    if self.add_input_representations:
                        layer_embed += mask_embed
                        layer_feat += mask_feature
                    if self.fuse:
                        embed_matrix = (layer_embed @ layer_embed.T).unsqueeze(-1)
                        feat_matrix = (layer_feat @ layer_feat.T).unsqueeze(-1)
                        matrix = torch.cat((embed_matrix, feat_matrix), dim=-1)
                    elif not self.transpose_queries:
                        matrix = (layer_embed @ layer_feat.T).unsqueeze(-1)
                    else:
                        matrix = (layer_feat @ layer_embed.T).unsqueeze(-1)
                    pred = out_proj(matrix)  # [B, n, n, 1] --> [B, n, n, 2]
                    task_per_layer.append(pred)  # {1, 2}
                layer_preds.append(task_per_layer)
            preds = self.add_to_preds(preds, layer_preds)
        return preds  # [L, B, n, n, 2]

    def pool_features(self, features: List[torch.Tensor]):
        if self.attn_before_pool:
            for i, sample in enumerate(features):
                H, W = features[i].shape[-2:]

                if self.pos_enc:
                    features[i] = self.pe(features[i])
                feats = rearrange(features[i], "m c h w -> m (h w) c")
                if self.mask_non_mask_values:
                    attn_mask = torch.ones_like(features[i], dtype=torch.bool)
                    attn_mask[torch.where(features[i] != 0)] = False
                    num_heads = self.features_transformer.layers[0].attn.num_heads
                    attn_mask = rearrange(
                        attn_mask, "m (n c) h w -> (m n) (h w) c", n=num_heads
                    )
                if self.learnable_pe:
                    feats += self.pe.weight
                feats = self.features_transformer(
                    feats,
                    return_intermediate=False,
                    attn_mask=attn_mask if self.mask_non_mask_values else None,
                )
                feats = rearrange(feats, "m (h w) c -> m c h w", h=H, w=W)

                # TODO: pool more properly by just getting the elements needed
                if self.mask_non_mask_values:
                    feats[torch.where(sample == 0)] = 0  # masking non mask values
                features[i] = feats
        pooled_features = [
            rearrange(
                self.pooling(feature, feature.shape[2:]),
                "n c ... -> n (c ...)",
            )
            for feature in features
        ]
        return pooled_features

    def forward(self, mask_embeddings, mask_features):
        return self.forward_non_uniform(mask_embeddings, mask_features)


@GEOMETRIC_PREDICTOR_REGISTRY.register()
class SingleTransformer(nn.Module):
    @configurable
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        geometric_tasks: str,
        mlp_num_layers: int,
        expansion: int,
        queries_inputs: bool,
        pooling_type=F.max_pool2d,
    ):
        super().__init__()

        self.task_names = []
        if "o" in geometric_tasks:
            self.task_names.append("pred_occlusion")
        if "d" in geometric_tasks:
            self.task_names.append("pred_depth")

        self.queries_inputs = queries_inputs
        self.pooling = pooling_type
        self.num_layers = num_layers
        self.transformer = Transformer(d_model, num_heads, num_layers, mlp_num_layers)

        self.projs = nn.ModuleList([])
        for _ in self.task_names:
            self.projs.append(
                MLP(d_model, d_model * expansion, d_model, mlp_num_layers)
            )

        self.out_projs = nn.ModuleList([])
        for _ in self.task_names:
            self.out_projs.append(MLP(1, d_model, 2, mlp_num_layers))

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_preds_dict(self):
        preds = defaultdict(list)
        for task in self.task_names:
            preds[task] = [[] for _ in range(self.num_layers)]
        return preds

    def add_to_preds(self, preds, layer_preds):
        for i, task in enumerate(self.task_names):
            for layer in range(self.num_layers):
                preds[task][layer].append(layer_preds[layer][i])
        return preds

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.GEOMETRIC_PREDICTOR.D_MODEL,
            "num_heads": cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_HEADS,
            "num_layers": cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_LAYERS,
            "geometric_tasks": cfg.MODEL.GEOMETRIC_PREDICTOR.TASK,
            "mlp_num_layers": cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_NUM_LAYERS,
            "expansion": cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_EXPANSION,
            "queries_inputs": cfg.MODEL.GEOMETRIC_PREDICTOR.QUERIES_INPUTS,
        }

    def forward(self, mask_embeddings, mask_features):
        preds = self.init_preds_dict()

        feature_embeddings = (
            mask_embeddings
            if self.queries_inputs
            else self.pool_features(mask_features)
        )
        for feat_embed in feature_embeddings:
            embeddings = self.transformer(feat_embed)  # list of embeds
            layer_preds = []
            for layer_embed in embeddings:
                task_per_layer = []
                for embed_mlp, out_proj in zip(self.projs, self.out_projs):
                    embed = embed_mlp(layer_embed)
                    matrix = (embed @ embed.T).unsqueeze(-1)
                    pred = out_proj(matrix)
                    task_per_layer.append(pred)  # {1, 2}
                layer_preds.append(task_per_layer)
            preds = self.add_to_preds(preds, layer_preds)
        return preds

    def pool_features(self, features: List[torch.Tensor]):
        return [
            rearrange(
                self.pooling(feature, feature.shape[2:]),
                "n c ... -> n (c ...)",
            )
            for feature in features
        ]


@GEOMETRIC_PREDICTOR_REGISTRY.register()
class CrossAttentionTransformer(nn.Module):
    @configurable
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        geometric_tasks: str,
        mlp_num_layers: int,
        expansion: int,
        queries_inputs: bool,
        pooling_type=F.max_pool2d,
        cross_attn: bool = True,
    ):
        super().__init__()

        self.task_names = []
        if "o" in geometric_tasks:
            self.task_names.append("pred_occlusion")
        if "d" in geometric_tasks:
            self.task_names.append("pred_depth")

        self.queries_inputs = queries_inputs
        self.pooling = pooling_type
        self.num_layers = num_layers
        self.transformer = Transformer(
            d_model, num_heads, num_layers, mlp_num_layers, cross_attn=cross_attn
        )

        self.projs = nn.ModuleList([])
        for _ in self.task_names:
            self.projs.append(
                MLP(d_model, d_model * expansion, d_model, mlp_num_layers)
            )

        self.out_projs = nn.ModuleList([])
        for _ in self.task_names:
            self.out_projs.append(MLP(1, d_model, 2, mlp_num_layers))

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_preds_dict(self):
        preds = defaultdict(list)
        for task in self.task_names:
            preds[task] = [[] for _ in range(self.num_layers)]
        return preds

    def add_to_preds(self, preds, layer_preds):
        for i, task in enumerate(self.task_names):
            for layer in range(self.num_layers):
                preds[task][layer].append(layer_preds[layer][i])
        return preds

    @classmethod
    def from_config(cls, cfg):
        return {
            "d_model": cfg.MODEL.GEOMETRIC_PREDICTOR.D_MODEL,
            "num_heads": cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_HEADS,
            "num_layers": cfg.MODEL.GEOMETRIC_PREDICTOR.NUM_LAYERS,
            "geometric_tasks": cfg.MODEL.GEOMETRIC_PREDICTOR.TASK,
            "mlp_num_layers": cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_NUM_LAYERS,
            "expansion": cfg.MODEL.GEOMETRIC_PREDICTOR.MLP_EXPANSION,
            "queries_inputs": cfg.MODEL.GEOMETRIC_PREDICTOR.QUERIES_INPUTS,
        }

    def forward(self, mask_embeddings, mask_features):
        preds = self.init_preds_dict()

        # TODO: switch for which features we use
        feature_embeddings = (
            mask_embeddings
            if self.queries_inputs
            else self.pool_features(mask_features)
        )
        cross_attn = (
            self.pool_features(mask_features)
            if self.queries_inputs
            else mask_embeddings
        )
        for feat_embed, ca in zip(feature_embeddings, cross_attn):
            embeddings = self.transformer(feat_embed, ca)  # list of embeds
            layer_preds = []
            for layer_embed in embeddings:
                task_per_layer = []
                for embed_mlp, out_proj in zip(self.projs, self.out_projs):
                    embed = embed_mlp(layer_embed)
                    matrix = (embed @ embed.T).unsqueeze(-1)
                    pred = out_proj(matrix)
                    task_per_layer.append(pred)  # {1, 2}
                layer_preds.append(task_per_layer)
            preds = self.add_to_preds(preds, layer_preds)
        return preds

    def pool_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            rearrange(
                self.pooling(feature, feature.shape[2:]),
                "n c ... -> n (c ...)",
            )
            for feature in features
        ]


def build_geometric_predictor(cfg):
    """
    Builds an occlusion predictor.
    """
    name = cfg.MODEL.GEOMETRIC_PREDICTOR.NAME
    if name == "":  # to run w/o occ head
        return None
    return GEOMETRIC_PREDICTOR_REGISTRY.get(name)(cfg)
