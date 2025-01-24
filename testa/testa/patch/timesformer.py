'''
Adapted from https://github.com/facebookresearch/ToMe
'''


from typing import Tuple
import torch.nn.functional as F
import torch
from models.timesformer.models.vit import Attention, Block, VisionTransformer
from einops import rearrange
from testa.merge import bipartite_soft_matching, merge_source, merge_wavg
from testa.merge_original import original_bipartite_soft_matching, original_merge_wavg
from testa.utils import parse_r, parse_merging_type


class TESTABlock(Block):
    """
    Modifications:
     - Apply TESTA between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, B, T, L) -> torch.Tensor:
        """
        x: [bsz, 1+seq_len*n_frm, dim] for video
        """
        attn_size = self._testa_info["size"] if self._testa_info["prop_attn"] else None
        merging_type = self._testa_info["merging_type"].pop(0)
        if self.attention_type in ['space_only', 'joint_space_time']:
            x = self.global_agg(x)
        elif self.attention_type == 'divided_space_time':
            # Temporal
            xt = x[:, 1:, :]  # [B, LxT, D]
            xt = rearrange(xt, 'b (l t) d -> (b l) t d', b=B, l=L, t=T)
            xt_attn, metric_t = self.temporal_attn(self.temporal_norm1(xt))
            if self.learnable_temporal_scaling is False:
                res_temporal = self.drop_path(xt_attn)
            else:
                res_temporal = self.drop_path(xt_attn * (torch.tanh(self.temporal_scaling) + 1))
            res_temporal = rearrange(res_temporal, '(b l) t d -> b (l t) d', b=B, l=L, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal
            if 'frame' in merging_type:
                xt = self.testa(xt, metric_t, B, L, 'frame')
                # reconstruct
                T = xt.size(1) // L

            # Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)  # [B, 1, D]
            cls_token = init_cls_token.repeat(1, T, 1)  # [B, T, D]
            cls_token = rearrange(cls_token, 'b t d -> (b t) d', b=B, t=T).unsqueeze(1)  # [BxT, 1, D]
            xs = xt  # [B, LxT, D]
            xs = rearrange(xs, 'b (l t) d -> (b t) l d', b=B, l=L, t=T)
            xs = torch.cat((cls_token, xs), 1)  # [BxT, 1+L, D]
            x_attn, metric_s = self.attn(self.norm1(xs), attn_size)  # cal metric for TESTA
            res_spatial = self.drop_path(x_attn)

            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]  # [BxT, 1, D]
            cls_token = rearrange(cls_token, '(b t) d -> b t d', b=B, t=T)  # [B, T, D]
            cls_token = torch.mean(cls_token, 1, True)  # averaging for every frame  [B, 1, D]
            res_spatial = res_spatial[:, 1:, :]  # [BxT, L, D]
            res_spatial = rearrange(res_spatial, '(b t) l d -> b (l t) d', b=B, l=L, t=T)
            res = res_spatial  # [B, LxT, D]
            x = xt  # [B, LxT, D], feature before spatial attn

            # Mlp
            x = rearrange((x + res), 'b (l t) d -> (b t) l d', b=B, l=L, t=T)  # [BxT, L, D]
            final_cls = init_cls_token + cls_token
            x = torch.cat((final_cls.repeat(x.size(0) // final_cls.size(0), 1, 1), x), 1)
            if 'patch' in merging_type:
                x = self.testa(x, metric_s, B, L, 'patch')[:, 1:, :]  # exclude [cls]
            # reconstruct
            L = x.size(1)
            x = rearrange(x, '(b t) l d -> b (l t) d', b=B, l=L, t=T)
            x = torch.cat((final_cls, x), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, T, L

    def global_agg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global aggregation of all patches in all frames
        """
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._testa_info["size"] if self._testa_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._testa_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = original_bipartite_soft_matching(
                metric,
                r,
                self._testa_info["class_token"],
                self._testa_info["distill_token"],
            )
            if self._testa_info["trace_source"]:
                self._testa_info["source"] = merge_source(
                    merge, x, self._testa_info["source"]
                )
            x, self._testa_info["size"] = original_merge_wavg(merge, x, self._testa_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

    def testa(self, x, metric, B, L, merging_type):
        r = self._testa_info["r"].pop(0)
        if r > 0:
            if merging_type == 'patch':
                x = rearrange(x, "(b t) l d -> b t l d", b=B)
                metric = rearrange(metric, "(b t) l d -> b t l d", b=B)
            else:  # merging_type == 'frame'
                x = rearrange(x, "b (l t) d -> b l t d", l=L)
                metric = rearrange(metric, "(b l) t d -> b l t d", b=B)
                if self._testa_info["size"] is not None:
                    # by default, the size of self._testa_info["size"] is [b, t, l, d]
                    self._testa_info["size"] = self._testa_info["size"].permute(0, 2, 1, 3)
                    self._testa_info["size"] = self._testa_info["size"][:, 1:, ...]  # remove cls
            # Apply TESTA here
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
            if merging_type == 'patch':
                x = rearrange(x, "b t l d -> (b t) l d", b=B)
            else:  # merging_type == 'frame'
                self._testa_info["size"] = self._testa_info["size"].permute(0, 2, 1, 3)
                size_cls = torch.ones(B, self._testa_info["size"].size(1), 1, 1).to(self._testa_info["size"])
                self._testa_info["size"] = torch.cat([size_cls, self._testa_info["size"]], dim=-2)  # add cls
                x = rearrange(x, "b l t d -> b (l t) d", l=L)
        return x


class TESTAAttention(Attention):
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
    class TESTAVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward_features(self, x, get_all_tokens=True):
            B = x.shape[0]
            x, T, W = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # resizing the positional embeddings in case they don't match the input at inference
            if x.size(1) != self.pos_embed.size(1):
                pos_embed = self.pos_embed
                cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                H = x.size(1) // W
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            # Time Embeddings
            if self.attention_type != 'space_only':
                cls_tokens = x[:B, 0, :].unsqueeze(1)
                x = x[:, 1:]
                x = rearrange(x, '(b t) l d -> (b l) t d', b=B, t=T)
                # Resizing time embeddings in case they don't match
                if T != self.time_embed.size(1):
                    time_embed = self.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                    new_time_embed = new_time_embed.transpose(1, 2)
                    x = x + new_time_embed
                else:
                    x = x + self.time_embed
                x = self.time_drop(x)
                x = rearrange(x, '(b l) t d -> b (l t) d', b=B, t=T)
                x = torch.cat((cls_tokens, x), dim=1)

            # Attention blocks
            L = (x.size(1) - 1) // T
            for blk in self.blocks:
                x, T, L = blk(x, B, T, L)

            # Predictions for space-only baseline
            if self.attention_type == 'space_only':
                x = rearrange(x, '(b t) l d -> b t l d', b=B, t=T)
                if get_all_tokens is False:
                    x = torch.mean(x, 1)  # averaging predictions for every frame
                else:
                    x = self.norm(x)
                    x = rearrange(x, 'b t l d -> b (t l) d', b=B, t=T)  # concat tokens of every frame
                    return x
            x = self.norm(x)
            if get_all_tokens is False:
                return x[:0]
            else:
                return x

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            r = self.r.copy() if isinstance(self.r, list) else self.r
            merging_type = self.merging_type.copy() if isinstance(self.merging_type, list) else self.merging_type
            self._testa_info["r"] = parse_r(len(self.blocks), r)
            self._testa_info["merging_type"] = parse_merging_type(len(self.blocks), merging_type)
            self._testa_info["size"] = None
            self._testa_info["source"] = None

            return super().forward(*args, **kwdargs)

    return TESTAVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, merging_type: str = 'patch', num_patches: int = 196
):
    """
    Applies TESTA to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._testa_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    TESTAVisionTransformer = make_testa_class(model.__class__)

    model.__class__ = TESTAVisionTransformer
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
        "num_patches": num_patches,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._testa_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TESTABlock
            module._testa_info = model._testa_info
        elif isinstance(module, Attention):
            module.__class__ = TESTAAttention
