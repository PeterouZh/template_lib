import yaml
import math
from easydict import EasyDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers_v2.build import D2LAYERv2_REGISTRY, build_d2layer_v2
from template_lib.utils import get_attr_kwargs, update_config




@D2LAYERv2_REGISTRY.register()
class Conv2d(nn.Conv2d):
  """
  """
  def __init__(self, cfg, **kwargs):

    # fmt: off
    in_channels                   = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    out_channels                  = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    padding                       = get_attr_kwargs(cfg, 'padding', default=0, **kwargs)
    dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    groups                        = get_attr_kwargs(cfg, 'groups', default=1, **kwargs)
    bias                          = get_attr_kwargs(cfg, 'bias', default=True, **kwargs)
    padding_mode                  = get_attr_kwargs(cfg, 'padding_mode', default='zeros', **kwargs)
    # fmt: on

    super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)

  def forward(self, input, **kargs):
    x = super(Conv2d, self).forward(input)
    return x


@D2LAYERv2_REGISTRY.register()
class ModulatedConv2d(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    # fmt: off
    self.in_channels               = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels              = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.kernel_size               = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.use_affine                = get_attr_kwargs(cfg, 'use_affine', default=False, **kwargs)
    self.style_dim                 = get_attr_kwargs(cfg, 'style_dim', default=None, **kwargs)
    self.demodulate                = get_attr_kwargs(cfg, 'demodulate', default=True, **kwargs)
    # fmt: on

    fan_in = self.in_channels * self.kernel_size ** 2
    self.scale = 1 / math.sqrt(fan_in)
    self.padding = self.kernel_size // 2

    self.weight = nn.Parameter(torch.randn(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
    if self.use_affine:
      self.modulation = nn.Linear(self.style_dim, self.in_channels)
    pass

  def forward(self, input, style):
    batch, in_channel, height, width = input.shape

    if self.use_affine:
      style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
    else:
      style = style.view(batch, 1, in_channel, 1, 1)
    weight = self.scale * self.weight * style

    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
      weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

    weight = weight.view(batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size)

    input = input.view(1, batch * in_channel, height, width)
    out = F.conv2d(input, weight, padding=self.padding, groups=batch)
    _, _, height, width = out.shape
    out = out.view(batch, self.out_channels, height, width)

    return out

  def __repr__(self):
    return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.kernel_size})')

  @staticmethod
  def test_case():
    import template_lib.d2.layers_v2.convs

    # test use_affine=false
    cfg_str = """
              name: "ModulatedConv2d"
              update_cfg: true
              in_channels: 128
              out_channels: 128
              use_affine: false
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = ModulatedConv2d.update_cfg(cfg)

    op = build_d2layer_v2(cfg)
    op.cuda()

    bs = 2
    x = torch.randn(bs, op.in_channels, 8, 8).cuda()
    style = torch.randn(bs, op.in_channels).cuda()
    out = op(x, style)

    # test use_affine=true
    cfg_str = """
                  name: "ModulatedConv2d"
                  update_cfg: true
                  in_channels: 128
                  out_channels: 128
                  use_affine: true
                  style_dim: 256
                  """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = ModulatedConv2d.update_cfg(cfg)

    op = build_d2layer_v2(cfg)
    op.cuda()

    bs = 2
    x = torch.randn(bs, op.in_channels, 8, 8).cuda()
    style = torch.randn(bs, op.style_dim).cuda()
    out = op(x, style)
    pass

  @staticmethod
  def update_cfg(cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "ModulatedConv2d"
      in_channels: 128
      out_channels: 128
      kernel_size: 3      
      use_affine: true
      style_dim: 256      
      demodulate: true
      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg


@D2LAYERv2_REGISTRY.register()
class EmbeddingModulatedConv2d(nn.Module):
  def __init__(self, cfg, **kwargs):
    super().__init__()

    # fmt: off
    self.n_classes                 = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    self.style_dim                 = get_attr_kwargs(cfg, 'style_dim', default=None, **kwargs)
    self.ModulatedConv2d_cfg       = get_attr_kwargs(cfg, 'ModulatedConv2d_cfg', default=None, **kwargs)
    # fmt: on

    self.embedding = nn.Embedding(self.n_classes, self.style_dim)
    self.mod_conv = build_d2layer_v2(self.ModulatedConv2d_cfg, **kwargs)
    pass

  def forward(self, input, y):
    """
    y: class label
    """
    y = y.squeeze()
    style = self.embedding(y)
    out = self.mod_conv(input, style)
    return out

  @staticmethod
  def test_case():
    import template_lib.d2.layers_v2.convs

    # test use_affine=false
    cfg_str = """
              name: "EmbeddingModulatedConv2d"
              n_classes: 10
              style_dim: 128
              ModulatedConv2d_cfg:
                name: "ModulatedConv2d"
                in_channels: 128
                out_channels: 128
                kernel_size: 3      
                use_affine: false
                demodulate: true
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = ModulatedConv2d.update_cfg(cfg)

    op = build_d2layer_v2(cfg)
    op.cuda()

    bs = 2
    in_channels = 128
    x = torch.randn(bs, in_channels, 8, 8).cuda()
    y = torch.tensor([0, 1]).view(bs, -1).cuda()
    out = op(x, y)

    pass

  @staticmethod
  def update_cfg(cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
        name: "ModulatedConv2d"
        n_classes: 10
        style_dim: 128
        ModulatedConv2d_cfg:
          name: "ModulatedConv2d"
          in_channels: 128
          out_channels: 128
          kernel_size: 3      
          use_affine: false
          style_dim: 256      
          demodulate: true
        """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg
