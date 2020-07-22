import math
import yaml
from easydict import EasyDict
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from template_lib.d2.layers.build import D2LAYER_REGISTRY, build_d2layer
from template_lib.utils import get_attr_kwargs, update_config


@D2LAYER_REGISTRY.register()
class DenseBlockMultiPath(nn.Module):

  def __init__(self, cfg, **kwargs):
    super().__init__()

    cfg = self.update_cfg(cfg)

    self.in_channels           = get_attr_kwargs(cfg, 'in_channels', **kwargs)
    self.n_nodes               = get_attr_kwargs(cfg, 'n_nodes', **kwargs)
    self.cfg_mix_layer         = get_attr_kwargs(cfg, 'cfg_mix_layer', **kwargs)
    self.cfg_ops               = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cell_op_idx           = get_attr_kwargs(cfg, 'cell_op_idx', default=None, **kwargs)

    self.num_edges = self.get_edges(self.n_nodes)
    self.cfg_keys = list(self.cfg_ops.keys())
    self.out_channels = self.in_channels

    assert (self.in_channels) % self.n_nodes == 0
    self.internal_c = self.in_channels // self.n_nodes

    self.in_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.internal_c,
                             kernel_size=1, stride=1, padding=0)

    # generate dag
    edge_idx = 0
    self.dag = nn.ModuleList()
    for i in range(self.n_nodes):
      self.dag.append(nn.ModuleList())
      for j in range(1 + i):
        if self.cell_op_idx is not None:
          op_key = self.cfg_keys[self.cell_op_idx[edge_idx]]
          cfg_ops = EasyDict({op_key: self.cfg_ops[op_key]})
          edge_idx += 1
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : cfg_ops})
        else:
          op = build_d2layer(self.cfg_mix_layer, **{**kwargs, "in_channels": self.internal_c,
                                                    "out_channels"         : self.internal_c,
                                                    "cfg_ops"              : self.cfg_ops})
        self.dag[i].append(op)
    pass

  def forward(self, x, batched_arcs, **kwargs):
    skip = x
    x = self.in_conv(x)

    states = [x, ]
    idx_start = 0
    for edges in self.dag:
      edges_arcs = batched_arcs[:, idx_start:idx_start + len(edges)]
      idx_start += len(edges)
      s_cur = sum(edge(s, edges_arcs[:, i], **kwargs) for i, (edge, s) in enumerate(zip(edges, states)))

      states.append(s_cur)

    x = torch.cat(states[1:], dim=1)
    x = x + skip
    return x

  @staticmethod
  def get_edges(n_nodes):
    num_edges = (n_nodes + 1) * n_nodes // 2
    return num_edges

  @staticmethod
  def test_case():
    cfg_str = """
              name: "DenseBlock"
              update_cfg: true
              in_channels: 144
              """
    cfg = EasyDict(yaml.safe_load(cfg_str))

    op = build_d2layer(cfg)
    op.cuda()

    bs = 2
    num_ops = len(op.cfg_ops)
    x = torch.randn(bs, 144, 8, 8).cuda()
    batched_arcs = torch.tensor([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]).cuda()
    out = op(x, batched_arcs)

    import torchviz
    g = torchviz.make_dot(out)
    g.view()
    pass

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "DenseCellCat"
      n_nodes: 4
      cfg_mix_layer:
        name: "MixedLayer"
      cfg_ops:
        Identity:
          name: "Identity"
        Conv2d_3x3:
          name: "Conv2dAct"
          cfg_act:
            name: "ReLU"
          cfg_conv:
            name: "Conv2d"
            kernel_size: 3
            padding: 1
        None:
          name: "D2None"

      """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg
