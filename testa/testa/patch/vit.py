'''
Adapted from https://github.com/facebookresearch/ToMe
'''

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from timm.models.layers import trunc_normal_, DropPath
from typing import Tuple
from einops import rearrange
import torch
from models.vit import Attention, Block, VisionTransformer, Mlp
import torch.nn as nn
from testa.merge import bipartite_soft_matching, merge_source, merge_wavg
from testa.utils import parse_r, parse_merging_type


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward(self, x, register_hook=False) -> torch.Tensor:
        """
        x: [bsz, seq_len, dim] for image; [bsz*n_frm, seq_len, dim] for video
        """
        attn_size = self._testa_info["size"] if self._testa_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)  # [frame, seq_len, dim], [frame, seq_len, dim/attn_head]
        x = x + self.drop_path(x_attn)

        r = self._testa_info["r"].pop(0)
        merging_type = self._testa_info["merging_type"].pop(0)
        if r > 0:
            x = rearrange(x, "(b t) l m -> b t l m", b=self._testa_info["B"])
            metric = rearrange(metric, "(b t) l m -> b t l m", b=self._testa_info["B"])
            # Apply ToMe here
            if merging_type == 'frame':
                x = x.permute(0, 2, 1, 3)
                metric = metric.permute(0, 2, 1, 3)
                if self._testa_info["size"] is not None:
                    self._testa_info["size"] = self._testa_info["size"].permute(0, 2, 1, 3)
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._testa_info["class_token"],
                self._testa_info["distill_token"],
                merging_type,
            )
            if self._testa_info["trace_source"]:
                self._testa_info["source"] = merge_source(
                    merge, x, self._testa_info["source"]
                )
            x, self._testa_info["size"] = merge_wavg(merge, x, self._testa_info["size"])
            if merging_type == 'frame':
                x = x.permute(0, 2, 1, 3)
                self._testa_info["size"] = self._testa_info["size"].permute(0, 2, 1, 3)
            B, T, L, M = x.size()
            x = x.reshape(-1, L, M)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_testa_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1) -> torch.Tensor:
            r = self.r.copy() if isinstance(self.r, list) else self.r
            merging_type = self.merging_type.copy() if isinstance(self.merging_type, list) else self.merging_type
            self._testa_info["r"] = parse_r(len(self.blocks), r)
            self._testa_info["merging_type"] = parse_merging_type(len(self.blocks), merging_type)
            self._testa_info["size"] = None
            self._testa_info["source"] = None

            B, T = x.shape[0], x.shape[1]
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            self._testa_info["B"] = B

            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B*T, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            x = x + self.pos_embed[:, :x.size(1), :]
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                x = blk(x, register_blk == i)
            x = self.norm(x)

            return x

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, merging_type: str = 'patch',
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._testa_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_testa_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model.merging_type = merging_type
    model._testa_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None and 'patch' in merging_type,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._testa_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._testa_info = model._testa_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
