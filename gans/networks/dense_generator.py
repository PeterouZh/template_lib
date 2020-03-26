import json
from easydict import EasyDict
import functools
import torch
from torch import nn

from template_lib.utils import get_attr_kwargs
from template_lib.d2.layers import build_d2layer
from template_lib.d2.utils import comm

from .build import DISCRIMINATOR_REGISTRY, GENERATOR_REGISTRY


@GENERATOR_REGISTRY.register()
class DenseGenerator_v1(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(DenseGenerator_v1, self).__init__()

    self.ch                            = get_attr_kwargs(cfg, 'ch', default=512, **kwargs)
    self.linear_ch                     = get_attr_kwargs(cfg, 'linear_ch', default=128, **kwargs)
    self.bottom_width                  = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
    self.dim_z                         = get_attr_kwargs(cfg, 'dim_z', default=128, **kwargs)
    self.init_type                     = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
    self.cfg_upsample                  = get_attr_kwargs(cfg, 'cfg_upsample', **kwargs)
    self.num_cells                     = get_attr_kwargs(cfg, 'num_cells', **kwargs)
    self.cfg_cell                      = get_attr_kwargs(cfg, 'cfg_cell', **kwargs)
    self.layer_op_idx                  = get_attr_kwargs(cfg, 'layer_op_idx', default=None, **kwargs)
    self.cfg_ops                       = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cfg_out_bn                    = get_attr_kwargs(cfg, 'cfg_out_bn', **kwargs)

    if self.layer_op_idx is not None:
      self.layer_op_idx = json.loads(self.layer_op_idx.replace(' ', ', ').replace('][', '], [').strip())
      assert len(self.layer_op_idx) % self.num_cells == 0
      num_edges_of_cell = len(self.layer_op_idx) // self.num_cells

    self.device = torch.device(f'cuda:{comm.get_rank()}')

    self.l1 = nn.Linear(self.dim_z, (self.bottom_width ** 2) * self.linear_ch)
    self.conv1 = nn.Conv2d(in_channels=self.linear_ch, out_channels=self.ch,
                           kernel_size=1, stride=1, padding=0)
    self.upsample = build_d2layer(self.cfg_upsample)

    self.cells = nn.ModuleList()
    for i in range(self.num_cells):
      if self.layer_op_idx is not None:
        cell = build_d2layer(self.cfg_cell, in_channels=self.ch, cfg_ops=self.cfg_ops,
                             cell_op_idx=self.layer_op_idx[i * num_edges_of_cell:(i + 1) * num_edges_of_cell])
      else:
        cell = build_d2layer(self.cfg_cell, in_channels=self.ch, cfg_ops=self.cfg_ops)
      self.cells.append(cell)

    self.num_branches = len(self.cfg_ops)
    self.num_edges_of_cell = cell.num_edges
    self.num_layers = len(self.cells) * self.num_edges_of_cell

    out_bn = build_d2layer(self.cfg_out_bn, num_features=self.ch, **kwargs)
    self.to_rgb = nn.Sequential(
      out_bn,
      nn.ReLU(),
      nn.Conv2d(self.ch, 3, 3, 1, 1),
      nn.Tanh()
    )

    weights_init_func = functools.partial(self.weights_init, init_type=self.init_type)
    self.apply(weights_init_func)

  def forward(self, z, batched_arcs, *args, **kwargs):
    batched_arcs = batched_arcs.to(self.device)
    z = z.to(self.device)

    h = self.l1(z).view(-1, self.linear_ch, self.bottom_width, self.bottom_width)
    h = self.conv1(h)

    for idx, cell in enumerate(self.cells):
      h = self.upsample(h)
      h = cell(h, batched_arcs=batched_arcs[:, idx * self.num_edges_of_cell:(idx + 1) * self.num_edges_of_cell])

    output = self.to_rgb(h)

    return output

  @staticmethod
  def weights_init(m, init_type='orth'):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
      if init_type == 'normal':
        nn.init.normal_(m.weight.data, 0.0, 0.02)
      elif init_type == 'orth':
        nn.init.orthogonal_(m.weight.data)
      elif init_type == 'xavier_uniform':
        nn.init.xavier_uniform(m.weight.data, 1.)
      else:
        raise NotImplementedError(
          '{} unknown inital type'.format(init_type))
    elif (isinstance(m, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0.0)


@GENERATOR_REGISTRY.register()
class DenseGenerator_v2(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(DenseGenerator_v2, self).__init__()

    self.ch = get_attr_kwargs(cfg, 'ch', default=256, **kwargs)
    self.linear_ch = get_attr_kwargs(cfg, 'linear_ch', default=128, **kwargs)
    self.bottom_width = get_attr_kwargs(cfg, 'bottom_width', default=4, **kwargs)
    self.dim_z = get_attr_kwargs(cfg, 'dim_z', default=128, **kwargs)
    self.init_type = get_attr_kwargs(cfg, 'init_type', default='xavier_uniform', **kwargs)
    self.cfg_upsample = get_attr_kwargs(cfg, 'cfg_upsample', **kwargs)
    self.num_cells = get_attr_kwargs(cfg, 'num_cells', **kwargs)
    self.cfg_cell = get_attr_kwargs(cfg, 'cfg_cell', **kwargs)
    self.layer_op_idx = get_attr_kwargs(cfg, 'layer_op_idx', default=None, **kwargs)
    self.cfg_ops = get_attr_kwargs(cfg, 'cfg_ops', **kwargs)
    self.cfg_out_bn = get_attr_kwargs(cfg, 'cfg_out_bn', **kwargs)

    if self.layer_op_idx is not None:
      self.layer_op_idx = json.loads(self.layer_op_idx.replace(' ', ', ').replace('][', '], [').strip())
      assert len(self.layer_op_idx) % self.num_cells == 0
      num_edges_of_cell = len(self.layer_op_idx) // self.num_cells

    self.device = torch.device(f'cuda:{comm.get_rank()}')

    self.l1 = nn.Linear(self.dim_z, (self.bottom_width ** 2) * self.linear_ch)

    # self.act = nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels=self.linear_ch, out_channels=self.ch,
                           kernel_size=1, stride=1, padding=0)

    self.upsample = build_d2layer(self.cfg_upsample)

    self.cells = nn.ModuleList()
    for i in range(self.num_cells):
      if self.layer_op_idx is not None:
        cell = build_d2layer(self.cfg_cell, in_channels=self.ch, cfg_ops=self.cfg_ops,
                             cell_op_idx=self.layer_op_idx[i * num_edges_of_cell:(i + 1) * num_edges_of_cell])
      else:
        cell = build_d2layer(self.cfg_cell, in_channels=self.ch, cfg_ops=self.cfg_ops)
      self.cells.append(cell)

    self.num_branches = len(self.cfg_ops)
    self.num_edges_of_cell = cell.num_edges
    self.num_layers = len(self.cells) * self.num_edges_of_cell

    out_bn = build_d2layer(self.cfg_out_bn, num_features=self.ch, **kwargs)
    self.to_rgb = nn.Sequential(
      out_bn,
      nn.ReLU(),
      nn.Conv2d(self.ch, 3, 3, 1, 1),
      nn.Tanh()
    )

    weights_init_func = functools.partial(self.weights_init, init_type=self.init_type)
    self.apply(weights_init_func)

  def forward(self, z, batched_arcs, *args, **kwargs):
    batched_arcs = batched_arcs.to(self.device)
    z = z.to(self.device)

    h = self.l1(z).view(-1, self.linear_ch, self.bottom_width, self.bottom_width)
    # h = self.act(h)
    h = self.conv1(h)

    for idx, cell in enumerate(self.cells):
      h = self.upsample(h)
      h = cell(h, batched_arcs=batched_arcs[:, idx * self.num_edges_of_cell:(idx + 1) * self.num_edges_of_cell])

    output = self.to_rgb(h)

    return output

  @staticmethod
  def weights_init(m, init_type='orth'):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
      if init_type == 'normal':
        nn.init.normal_(m.weight.data, 0.0, 0.02)
      elif init_type == 'orth':
        nn.init.orthogonal_(m.weight.data)
      elif init_type == 'xavier_uniform':
        nn.init.xavier_uniform(m.weight.data, 1.)
      else:
        raise NotImplementedError(
          '{} unknown inital type'.format(init_type))
    elif (isinstance(m, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0.0)



