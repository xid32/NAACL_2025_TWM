'''
Adapted from https://github.com/facebookresearch/ToMe
'''

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    merging_type: str = 'patch'
) -> Tuple[Callable, Callable]:
    """
    Applies TESTA with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[-2]  # dimension for reduction
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)  # |Set A| * |Set B| edges

        if merging_type == 'patch' and class_token:
            scores[..., 0, :] = -math.inf
        if merging_type == 'patch' and distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)  # keep edge with the highest sim for every node in Set A

        if merging_type == 'frame':  # aggregate frames based on patch voting
            n = metric.size(-3)  # number of patches
            node_idx, _ = node_idx.mode(dim=-2, keepdim=True)
            node_idx = node_idx.repeat(1, n, 1)
            node_max, _ = node_max.mode(dim=-2, keepdim=True)
            node_max = node_max.repeat(1, n, 1)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # sort |Set A| edges based on sim

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # node_idx: idx for Set B

        if class_token or merging_type == 'frame':
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=-2)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        B, n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(B, n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(B, n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[-2]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        B, n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(B, n, r, c))

        out = torch.zeros(B, n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(B, n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(B, n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        B, n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(B, n, t, t)

    source = merge(source, mode="amax")
    return source
