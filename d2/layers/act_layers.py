import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY
from template_lib.utils import get_attr_kwargs


@D2LAYER_REGISTRY.register()
class ReLU(nn.ReLU):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    self.inplace               = getattr(cfg, 'inplace', False)

    super(ReLU, self).__init__(inplace=self.inplace)



@D2LAYER_REGISTRY.register()
class NoAct(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

  def forward(self, x):
    return x