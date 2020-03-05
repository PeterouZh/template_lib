import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY
from template_lib.utils import get_attr_kwargs

from .pagan_layers_utils import SN


@D2LAYER_REGISTRY.register()
class SNConv2d(nn.Conv2d, SN):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    self.in_channels                   = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels                  = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    self.padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    self.dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    self.groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)
    self.bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    self.num_svs                       = get_attr_kwargs(cfg, 'num_svs', default=1, **kwargs)
    self.num_itrs                      = get_attr_kwargs(cfg, 'num_itrs', default=1, **kwargs)
    self.eps                           = get_attr_kwargs(cfg, 'eps', default=1e-6, **kwargs)

    nn.Conv2d.__init__(self, self.in_channels, self.out_channels, self.kernel_size, self.stride,
                       self.padding, self.dilation, self.groups, self.bias)
    SN.__init__(self, self.num_svs, self.num_itrs, self.out_channels, eps=self.eps)

  def forward(self, x, *args):
    x = F.conv2d(x, self.W_(), self.bias, self.stride, self.padding, self.dilation, self.groups)
    return x


@D2LAYER_REGISTRY.register()
class Conv2d(nn.Conv2d):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    self.in_channels                   = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels                  = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    self.padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    self.dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    self.groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)
    self.bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    self.padding_mode                  = get_attr_kwargs(cfg, 'padding_mode', default='zeros', **kwargs)


    super(Conv2d, self).__init__(in_channels=self.in_channels, out_channels=self.out_channels,
                                 kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                 dilation=self.dilation, groups=self.groups, bias=self.bias,
                                 padding_mode=self.padding_mode)

  def forward(self, input, *args):
    x = super(Conv2d, self).forward(input)
    return x


@D2LAYER_REGISTRY.register()
class Linear(nn.Linear):
  """
  # 2D Conv layer with spectral norm
  """
  def __init__(self, cfg, **kwargs):

    in_features                   = get_attr_kwargs(cfg, 'in_features', **kwargs)
    out_features                  = get_attr_kwargs(cfg, 'out_features', **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)

    super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

