'''
Adapted from https://github.com/facebookresearch/ToMe
'''

from .vit import apply_patch as vit
from .timesformer import apply_patch as timesformer

__all__ = ["vit", "timesformer"]
