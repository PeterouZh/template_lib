import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY
from template_lib.utils import get_attr_kwargs



@D2LAYER_REGISTRY.register()
class UpSample(nn.Module):

  def __init__(self, cfg, scale_factor=2, **kwargs):
    super(UpSample, self).__init__()

    self.scale_factor                   = get_attr_kwargs(cfg, 'scale_factor', default=2, **kwargs)
    self.mode                           = get_attr_kwargs(cfg, 'mode', default='bilinear', choices=['nearest'],
                                                          **kwargs)
    self.align_corners                  = get_attr_kwargs(cfg, 'align_corners', default=None, **kwargs)


  def forward(self, x):
    x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
    return x


@D2LAYER_REGISTRY.register()
class Identity(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

  def forward(self, x):
    return x