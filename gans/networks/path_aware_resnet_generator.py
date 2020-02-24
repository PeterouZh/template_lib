from collections import OrderedDict
from easydict import EasyDict
import yaml
import functools
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from template_lib.utils import get_eval_attr

from .pagan import layers
from .pagan.BigGAN import G_arch, D_arch
from .pagan.ops import \
  (MixedLayer, MixedLayerCond, UpSample, DownSample, Identity,
   MixedLayerSharedWeights, MixedLayerCondSharedWeights,
   SinglePathLayer)

from .build import GENERATOR_REGISTRY

__all__ = ['PathAwareResNetGen']


@GENERATOR_REGISTRY.register()
class PathAwareResNetGen(nn.Module):
  def __init__(self, cfg):
    super(PathAwareResNetGen, self).__init__()

    self.resolution                        = get_eval_attr(cfg.model.generator.resolution, {"cfg": cfg})
    self.ch                                = cfg.model.generator.ch
    self.attention                         = cfg.model.generator.attention
    self.use_sn                            = cfg.model.generator.use_sn
    self.dim_z                             = cfg.model.generator.dim_z
    self.bottom_width                      = cfg.model.generator.bottom_width
    self.track_running_stats               = cfg.model.generator.track_running_stats
    self.share_conv_weights                = cfg.model.generator.share_conv_weights
    self.single_path_layer                 = cfg.model.generator.single_path_layer
    self.share_bias                        = cfg.model.generator.share_bias
    self.output_type                       = cfg.model.generator.output_type
    self.bn_type                           = cfg.model.generator.bn_type
    self.ops                               = cfg.model.generator.ops
    self.init                              = cfg.model.generator.init
    self.use_sync_bn                       = getattr(cfg.model.generator, 'use_sync_bn', False)

    self.arch = G_arch(self.ch, self.attention)[self.resolution]

    if self.use_sn:
      self.which_linear = functools.partial(
        layers.SNLinear, num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv = functools.partial(
        layers.SNConv2d, kernel_size=3, padding=1,
        num_svs=1, num_itrs=1, eps=1e-6)
      self.which_conv_1x1 = functools.partial(
        layers.SNConv2d, kernel_size=1, padding=0,
        num_svs=1, num_itrs=1, eps=1e-6)
    else:
      self.which_linear = nn.Linear
      self.which_conv = functools.partial(
        nn.Conv2d, kernel_size=3, padding=1)
      self.which_conv_1x1 = functools.partial(
        nn.Conv2d, kernel_size=1, padding=0)

    # First linear layer
    self.linear = self.which_linear(
      self.dim_z, self.arch['in_channels'][0] * (self.bottom_width ** 2))

    num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * num_conv_in_block
    self.upsample_layer_idx = \
      [num_conv_in_block * l
        for l in range(0, self.num_layers//num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.layers_para_list = []
    self.skip_layers = nn.ModuleList([])
    bn_type = getattr(self, 'bn_type', 'bn').lower()

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//num_conv_in_block]
      if layer_id % num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out
      upsample = (UpSample()
                  if (self.arch['upsample'][layer_id//num_conv_in_block] and
                      layer_id in self.upsample_layer_idx)
                  else None)
      if getattr(self, 'share_conv_weights', False):
        if getattr(self, 'single_path_layer', False):
          layer = SinglePathLayer(
            layer_id=layer_id, in_planes=in_channels, out_planes=out_channels,
            ops=self.ops, track_running_stats=self.track_running_stats,
            scalesample=upsample, bn_type=bn_type, share_bias=self.share_bias)
        else:
          layer = MixedLayerSharedWeights(
            layer_id=layer_id, in_planes=in_channels, out_planes=out_channels,
            ops=self.ops, track_running_stats=self.track_running_stats,
            scalesample=upsample, bn_type=bn_type)
      else:
        layer = MixedLayer(
          layer_id, in_channels, out_channels,
          ops=self.ops, track_running_stats=self.track_running_stats,
          scalesample=upsample, bn_type=bn_type)
      self.layers.append(layer)
      self.layers_para_list.append(layer.num_para_list)
      if layer_id in self.upsample_layer_idx:
        skip_layers = []
        if self.arch['upsample'][layer_id//num_conv_in_block]:
          skip_layers.append(('upsample_%d'%layer_id, UpSample()))
        # if in_channels != out_channels:
          conv_1x1 = self.which_conv_1x1(in_channels, out_channels,
                                         kernel_size=1, padding=0)
          skip_layers.append(('upsample_%d_conv_1x1'%layer_id, conv_1x1))
        else:
          identity = Identity()
          skip_layers.append(('skip_%d_identity' % layer_id, identity))
        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    self.layers_para_matrix = np.array(self.layers_para_list).T
    # output layer
    self.output_type = getattr(self, 'output_type', 'snconv')
    self.output_sample_arc = False
    if self.output_type == 'snconv':
      if self.use_sync_bn:
        from detectron2.layers import NaiveSyncBatchNorm
        self.output_layer = nn.Sequential(
          NaiveSyncBatchNorm(self.arch['out_channels'][-1], affine=True, track_running_stats=self.track_running_stats),
          nn.ReLU(),
          self.which_conv(self.arch['out_channels'][-1], 3))
      else:
        self.output_layer = nn.Sequential(
          nn.BatchNorm2d(self.arch['out_channels'][-1], affine=True, track_running_stats=self.track_running_stats),
          nn.ReLU(),
          self.which_conv(self.arch['out_channels'][-1], 3))
    elif self.output_type == 'MixedLayer':
      self.output_sample_arc = True
      if getattr(self, 'share_conv_weights', False):
        self.output_conv = MixedLayerSharedWeights(
          layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
          track_running_stats=self.track_running_stats, scalesample=None,
          bn_type=bn_type)
      else:
        self.output_conv = MixedLayer(
          layer_id + 1, self.arch['out_channels'][-1], 3, ops=self.ops,
          track_running_stats=self.track_running_stats, scalesample=None,
          bn_type=bn_type)
    else:
      assert 0

    self.init_weights()
    pass

  def forward(self, x, sample_arcs):
    """

    :param x:
    :param sample_arcs: (b, num_layers)
    :return:
    """

    x = self.linear(x)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](x, sample_arc)

      if layer_id - 1 in self.upsample_layer_idx:
        x_up = self.skip_layers[upsample_layer](prev_layer)
        upsample_layer += 1
        x = x + x_up
        prev_layer = x

    if self.output_type == 'snconv':
      x = self.output_layer(x)
    elif self.output_type == 'MixedLayer':
      sample_arc = sample_arcs[:, -1]
      x = self.output_conv(x, sample_arc)
    else:
      assert 0
    x = torch.tanh(x)
    return x

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
              or isinstance(module, nn.Linear)
              or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
      if (isinstance(module, MixedLayerSharedWeights)):
        if self.init == 'ortho':
          for k, w in module.conv_weights.items():
            init.orthogonal_(w)
        else:
          assert 0
      if (isinstance(module, SinglePathLayer)):
        if self.init == 'ortho':
          init.orthogonal_(module.conv_weights_space)
        else:
          assert 0
      pass

