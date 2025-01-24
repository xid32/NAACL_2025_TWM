import logging
from collections import OrderedDict

import einops
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# WARNING: outdated. not usable.

def build_omnivore_swinb(config):
    """build omnivore swinb

    Args:
        config (dict): The config

    Returns: nn.Module.

    """
    from .omnivore_swin import CHECKPOINT_PATHS, SwinTransformer3D

    # new_wd = config.video_input.num_frames
    new_wd = config.swin_wd
    vision_encoder = SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(new_wd, 7, 7),
        drop_path_rate=0.1,  # TODO: set this based on the final models
        patch_norm=True,  # Make this the default value?
        depth_mode="summed_rgb_d_tokens",  # Make this the default value?
    )
    path = CHECKPOINT_PATHS[config.vit_name_or_pretrained_path]
    checkpoint = load_state_dict_from_url(path, progress=True, map_location="cpu")
    wd, wh, ww = 16, 7, 7

    # interpolate the rel_pos enmbedding.
    trunk_ckpt = checkpoint["trunk"]
    new_state_dict = OrderedDict()
    if new_wd != wd:
        for k, v in trunk_ckpt.items():
            if "relative_position_bias_table" in k:
                # do interpolation
                if config.pe_scale_method == "interpolation":
                    v = einops.rearrange(
                        v,
                        "(d h w c) nh -> nh c d h w",
                        d=2 * wd - 1,
                        h=2 * wh - 1,
                        w=2 * ww - 1,
                    )
                    v = F.interpolate(
                        v, size=(2 * new_wd - 1, 13, 13), mode="trilinear"
                    )  # shape: [nh, c, d, h, w]
                    v = einops.rearrange(v, "nh c d h w -> (d h w c) nh")
                elif config.pe_scale_method == "crop":
                    v = einops.rearrange(
                        v,
                        "(d h w c) nh -> d (h w c) nh",
                        d=2 * wd - 1,
                        h=2 * wh - 1,
                        w=2 * ww - 1,
                    )
                    v = v[wd - (new_wd) : wd + new_wd - 1]
                    v = einops.rearrange(
                        v,
                        "d (h w c) nh -> (d h w c) nh",
                        d=2 * new_wd - 1,
                        h=2 * wh - 1,
                        w=2 * ww - 1,
                    )
                else:
                    raise ValueError("not implemented")
            if "relative_position_index" not in k:
                new_state_dict[k] = v

    info = vision_encoder.load_state_dict(new_state_dict, strict=False)
    logger.info(f"SwinTransformer3D: loaded checkpoint {path}. info:{info}")
    return vision_encoder
