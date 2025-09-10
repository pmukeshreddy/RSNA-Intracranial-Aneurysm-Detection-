import torch
import torch.nn as nn
import torch.functional.nn.functional as F
from einops import rearrange
from typing import Sequence, Tuple, Optional
import numpy as np


class PatchEmbedding(nn.Module):
  def __init__(self,patch_size=4,in_chans=3,embed_dim=96,norm_layer=None,spatial_dims=3):
    super().__init__()
    self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * spatial_dims
    self.spatial_dims = spatial_dims

    if spatial_dims == 3:
      self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
    else:
      self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size[:2], stride=self.patch_size)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self,x):
    x = self.proj(x)
    if self.spatial_dims == 3:
            x = rearrange(x, 'b c d h w -> b (d h w) c')
    else:
            x = rearrange(x, 'b c h w -> b (h w) c')
    x = self.norm(x)
    return x
