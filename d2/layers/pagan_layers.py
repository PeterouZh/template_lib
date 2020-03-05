import torch
import torch.nn as nn
import torch.nn.functional as F


from template_lib.utils import get_attr_kwargs

from .build import D2LAYER_REGISTRY, build_d2layer


@D2LAYER_REGISTRY.register()
class MixedLayerCond(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(MixedLayerCond, self).__init__()

    self.in_channels                 = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.out_channels                = get_attr_kwargs(cfg, 'out_channels', **kwargs)
    self.cfg_ops                     = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)

    self.num_branch = len(self.cfg_ops)

    self.branches = nn.ModuleList()
    for name, cfg_op in self.cfg_ops.items():
      branch = build_d2layer(cfg_op, in_channels=self.in_channels, out_channels=self.out_channels)
      self.branches.append(branch)
    pass

  def forward(self, x, y, sample_arc):
    bs = len(sample_arc)
    sample_arc = sample_arc.type(torch.int64)

    sample_arc_onehot = torch.zeros(bs, self.num_branch).cuda()
    sample_arc_onehot[torch.arange(bs), sample_arc] = 1
    sample_arc_onehot = sample_arc_onehot.view(bs, self.num_branch, 1, 1, 1)

    x = [branch(x, y).unsqueeze(1) if idx in sample_arc else \
           (torch.zeros(bs, self.out_channels, x.size(-1), x.size(-1)).cuda().requires_grad_(False).unsqueeze(1)) \
         for idx, branch in enumerate(self.branches)]
    # x = [branch(x, y).unsqueeze(1) for branch in self.branches]
    x = torch.cat(x, 1)
    x = sample_arc_onehot * x
    x = x.sum(dim=1)

    return x


@D2LAYER_REGISTRY.register()
class NormActConv(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(NormActConv, self).__init__()

    self.C_in                          = get_attr_kwargs(cfg, 'C_in', **kwargs)
    self.C_out                         = get_attr_kwargs(cfg, 'C_out', **kwargs)
    self.kernel_size                   = get_attr_kwargs(cfg, 'kernel_size', **kwargs)
    self.stride                        = get_attr_kwargs(cfg, 'stride', default=1, **kwargs)
    self.padding                       = get_attr_kwargs(cfg, 'padding', default=1, **kwargs)
    self.dilation                      = get_attr_kwargs(cfg, 'dilation', default=1, **kwargs)
    self.weight                        = get_attr_kwargs(cfg, 'weight', default=None, **kwargs)
    self.bias                          = get_attr_kwargs(cfg, 'bias', default=None, **kwargs)
    self.which_act                     = get_attr_kwargs(cfg, 'which_act', default=nn.ReLU, **kwargs)
    self.sn_num_svs                    = getattr(cfg, 'sn_num_svs', 1)
    self.sn_num_itrs                   = getattr(cfg, 'sn_num_itrs', 1)
    self.sn_eps                        = getattr(cfg, 'sn_eps', 1e-6)
    self.cfg_bn                        = cfg.cfg_bn
    self.cfg_act                       = cfg.cfg_act
    self.cfg_conv                       = cfg.cfg_conv

    self.bn = build_d2layer(self.cfg_bn, num_features=self.C_in)
    self.act = build_d2layer(self.cfg_act)
    self.conv = build_d2layer(self.cfg_conv, )
    raise NotImplemented

  def forward(self, *inputs):
    x = self.bn(*inputs)
    x = self.act(x)
    x = self.conv(x)
    return x


