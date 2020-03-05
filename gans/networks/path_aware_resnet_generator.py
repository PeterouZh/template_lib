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

from template_lib.utils import get_attr_kwargs, update_config
from template_lib.d2.layers import build_d2layer

from .pagan import layers, build_layer
from .pagan.BigGAN import G_arch, D_arch
from .pagan.ops import \
  (MixedLayer, UpSample, DownSample, Identity,
   MixedLayerSharedWeights, MixedLayerCondSharedWeights,
   SinglePathLayer)

from .build import GENERATOR_REGISTRY

__all__ = ['PathAwareResNetGen']


@GENERATOR_REGISTRY.register()
class PathAwareResNetGen(nn.Module):
  def __init__(self, cfg, **kwargs):
    super(PathAwareResNetGen, self).__init__()

    self.img_size                          = get_attr_kwargs(cfg.model.generator, 'img_size', kwargs=kwargs)
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

    self.arch = G_arch(self.ch, self.attention)[self.img_size]

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

  def forward(self, x, sample_arcs, *args, **kwargs):
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


@GENERATOR_REGISTRY.register()
class PathAwareResNetGenCBN(nn.Module):

  def __init__(self, cfg, **kwargs):
    super(PathAwareResNetGenCBN, self).__init__()

    cfg = self.update_cfg(cfg)

    self.n_classes                 = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    self.img_size                  = get_attr_kwargs(cfg, 'img_size', **kwargs)
    self.ch                        = get_attr_kwargs(cfg, 'ch', default=8, **kwargs)
    self.attention                 = get_attr_kwargs(cfg, 'attention', default='0', **kwargs)
    self.dim_z                     = get_attr_kwargs(cfg, 'dim_z', default=256, **kwargs)
    self.hier                      = get_attr_kwargs(cfg, 'hier', default=True, **kwargs)
    self.embedding_dim             = get_attr_kwargs(cfg, 'embedding_dim', default=128, **kwargs)
    self.bottom_width              = get_attr_kwargs(cfg, 'bottom_width', **kwargs)
    self.init                      = get_attr_kwargs(cfg, 'init', default='ortho', **kwargs)
    self.cfg_first_fc              = cfg.cfg_first_fc
    self.cfg_bn                    = cfg.cfg_bn
    self.cfg_act                   = cfg.cfg_act
    self.cfg_mix_layer = cfg.cfg_mix_layer
    self.cfg_upsample              = cfg.cfg_upsample
    self.cfg_conv_1x1              = cfg.cfg_conv_1x1
    self.cfg_out_bn                = cfg.cfg_out_bn
    self.cfg_out_conv              = cfg.cfg_out_conv

    self.arch = G_arch(self.ch, self.attention)[self.img_size]

    if self.hier:
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      self.dim_z = self.z_chunk_size
      self.cbn_in_features = self.embedding_dim + self.z_chunk_size
    else:
      self.num_slots = 1
      self.z_chunk_size = 0
      self.cbn_in_features = self.embedding_dim

    # Prepare class embedding
    self.class_embedding = nn.Embedding(self.n_classes, self.embedding_dim)

    # First linear layer
    self.linear = build_d2layer(cfg.cfg_first_fc, in_features=self.dim_z,
                                out_features=self.arch['in_channels'][0] * (self.bottom_width ** 2))

    self.num_conv_in_block = 2
    self.num_layers = len(self.arch['in_channels']) * self.num_conv_in_block
    self.upsample_layer_idx = [self.num_conv_in_block * l for l in range(0, self.num_layers//self.num_conv_in_block)]

    self.layers = nn.ModuleList([])
    self.skip_layers = nn.ModuleList([])

    for layer_id in range(self.num_layers):
      block_in = self.arch['in_channels'][layer_id//self.num_conv_in_block]
      block_out = self.arch['out_channels'][layer_id//self.num_conv_in_block]
      if layer_id % self.num_conv_in_block == 0:
        in_channels = block_in
        out_channels = block_out
      else:
        in_channels = block_out
        out_channels = block_out

      # bn relu mix_layer upsample
      layer = []

      bn = build_d2layer(self.cfg_bn, in_features=self.cbn_in_features, out_features=in_channels)
      layer.append([f'bn_{layer_id}', bn])

      act = build_d2layer(self.cfg_act)
      layer.append([f'act_{layer_id}', act])

      mix_layer = build_d2layer(self.cfg_mix_layer, in_channels=in_channels, out_channels=out_channels, cfg_ops=cfg.cfg_ops)
      layer.append([f'mix_layer_{layer_id}', mix_layer])

      if layer_id in self.upsample_layer_idx:
        upsample = build_d2layer(self.cfg_upsample)
        layer.append([f'upsample_{layer_id}', upsample])

      layer = nn.Sequential(OrderedDict(layer))
      self.layers.append(layer)

      # skip branch
      if layer_id in self.upsample_layer_idx:
        skip_layers = []

        skip_conv_1x1 = build_d2layer(self.cfg_conv_1x1, in_channels=in_channels, out_channels=out_channels)
        skip_layers.append((f'skip_conv_1x1_{layer_id}', skip_conv_1x1))

        skip_upsample = build_d2layer(self.cfg_upsample)
        skip_layers.append((f'skip_upsample_{layer_id}', skip_upsample))

        skip_layers = nn.Sequential(OrderedDict(skip_layers))
        self.skip_layers.append(skip_layers)

    out_bn = build_d2layer(self.cfg_out_bn, num_features=self.arch['out_channels'][-1])
    out_act = build_d2layer(self.cfg_act)
    out_conv = build_d2layer(self.cfg_out_conv, in_channels=self.arch['out_channels'][-1])
    self.output_layer = nn.Sequential(OrderedDict([
      ('out_bn', out_bn),
      ('out_act', out_act),
      ('out_conv', out_conv)
    ]))

    self.init_weights()
    pass

  @staticmethod
  def update_cfg(cfg):
    if not getattr(cfg, 'update_cfg', False):
      return

    cfg_str = """
        name: "PathAwareResNetGenCBN"
        n_classes: "kwargs['n_classes']"
        img_size: "kwargs['img_size']"
        ch: 8
        dim_z: 256
        bottom_width: 4
        init: 'ortho'
        cfg_first_fc:
          name: "Linear"
          in_features: "kwargs['in_features']"
          out_features: "kwargs['out_features']"
        cfg_bn:
          name: "CondBatchNorm2d"
          in_features: "kwargs['in_features']"
          out_features: "kwargs['out_features']"
          cfg_fc:
            name: "Linear"
            in_features: "kwargs['in_features']"
            out_features: "kwargs['out_features']"
        cfg_act:
          name: "ReLU"
        cfg_mix_layer:
          name: "MixedLayerCond"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          cfg_ops: "kwargs['cfg_ops']"
        cfg_upsample:
          name: "UpSample"
          mode: "bilinear"
        cfg_conv_1x1:
          name: "Conv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: "kwargs['out_channels']"
          kernel_size: 1
        cfg_ops:
          SNConv2d_3x3:
            name: "SNConv2d"
            in_channels: "kwargs['in_channels']"
            out_channels: "kwargs['out_channels']"
            kernel_size: 3
            padding: 1
          Conv2d_3x3:
            name: "Conv2d"
            in_channels: "kwargs['in_channels']"
            out_channels: "kwargs['out_channels']"
            kernel_size: 3
            padding: 1
        cfg_out_bn:
          name: "BatchNorm2d"
          num_features: "kwargs['num_features']"
        cfg_out_conv:
          name: "Conv2d"
          in_channels: "kwargs['in_channels']"
          out_channels: 3
          kernel_size: 1
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def forward(self, z, y, sample_arcs):
    """

    :param sample_arcs: (b, num_layers)
    :return:
    """
    y = self.class_embedding(y)
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.arch['in_channels'])

    x = self.linear(z)
    x = x.view(x.size(0), -1, self.bottom_width, self.bottom_width)

    upsample_layer = 0
    for layer_id in range(self.num_layers):
      if layer_id == 0:
        prev_layer = x
      sample_arc = sample_arcs[:, layer_id]
      x = self.layers[layer_id](
        x=x, y=ys[layer_id // self.num_conv_in_block], sample_arc=sample_arc)

      if layer_id - 1 in self.upsample_layer_idx:
        x_up = self.skip_layers[upsample_layer](prev_layer)
        upsample_layer += 1
        x = x + x_up
        prev_layer = x

    x = self.output_layer(x)

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

      if (isinstance(module, (MixedLayerCondSharedWeights,
                              MixedLayerSharedWeights))):
        if self.init == 'ortho':
          for k, w in module.conv_weights.items():
            init.orthogonal_(w)
        else:
          assert 0