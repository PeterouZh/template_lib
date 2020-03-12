import tqdm
import collections
from collections import OrderedDict
from easydict import EasyDict
import yaml
import functools
import numpy as np

import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from template_lib.utils import get_attr_kwargs, update_config, get_ddp_attr, AverageMeter
from template_lib.d2.utils import comm
from template_lib.d2.layers import build_d2layer
from template_lib.d2.models.build import D2MODEL_REGISTRY
from template_lib.trainer.base_trainer import summary_defaultdict2txtfig


@D2MODEL_REGISTRY.register()
class FairControllerFC(nn.Module):
  '''
  https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
  '''

  def __init__(self, cfg, **kwargs):
    super(FairControllerFC, self).__init__()

    cfg = self.update_cfg(cfg)

    self.FID_IS                  = kwargs['FID_IS']
    self.myargs                  = kwargs['myargs']
    self.num_layers              = get_attr_kwargs(cfg, 'num_layers', **kwargs)
    self.num_branches            = get_attr_kwargs(cfg, 'num_branches', **kwargs)
    self.search_whole_channels   = get_attr_kwargs(cfg, 'search_whole_channels', default=True, **kwargs)
    self.lstm_size               = get_attr_kwargs(cfg, 'lstm_size', default=64, **kwargs)
    self.lstm_num_layers         = get_attr_kwargs(cfg, 'lstm_num_layers', default=2, **kwargs)
    self.tanh_constant           = get_attr_kwargs(cfg, 'tanh_constant', default=1.5, **kwargs)
    self.temperature             = get_attr_kwargs(cfg, 'temperature', default=-1, **kwargs)
    self.num_aggregate           = get_attr_kwargs(cfg, 'num_aggregate', **kwargs)
    self.entropy_weight          = get_attr_kwargs(cfg, 'entropy_weight', **kwargs)
    self.bl_dec                  = get_attr_kwargs(cfg, 'bl_dec', **kwargs)
    self.child_grad_bound        = get_attr_kwargs(cfg, 'child_grad_bound', **kwargs)
    self.log_every_iter          = get_attr_kwargs(cfg, 'log_every_iter', default=50, **kwargs)

    self.device = torch.device(f'cuda:{comm.get_rank()}')
    self.baseline = None
    self._create_params()
    self._reset_params()

  def update_cfg(self, cfg):
    if not getattr(cfg, 'update_cfg', False):
      return cfg

    cfg_str = """
      name: "PAGANRLController"
      num_layers: "kwargs['num_layers']"
      num_branches: "kwargs['num_branches']"
      lstm_size: 64
      lstm_num_layers: 2
      temperature: -1
      num_aggregate: 10
      entropy_weight: 0.0001
      bl_dec: 0.99
      child_grad_bound: 5.0
    """
    default_cfg = EasyDict(yaml.safe_load(cfg_str))
    cfg = update_config(default_cfg, cfg)
    return cfg

  def test_case(self):
    out = self()
    return out


  def _create_params(self):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
    '''
    self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                          hidden_size=self.lstm_size,
                          num_layers=self.lstm_num_layers)

    if self.search_whole_channels:
      self.w_emb = nn.Embedding(self.num_layers, self.lstm_size)
      self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=True)
    else:
      assert False, "Not implemented error: search_whole_channels = False"

    # self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    # self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

  def _reset_params(self):
    for m in self.modules():
      if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)

    nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
    nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

  def forward(self, bs, determine_sample=False):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
    '''
    h0 = None  # setting h0 to None will initialize LSTM state with 0s
    arc_seq = []
    entropys = []
    log_probs = []

    for layer_id in range(self.num_layers):
      if self.search_whole_channels:
        inputs = self.w_emb.weight[[layer_id]]
        inputs = inputs.unsqueeze(dim=0)
        output, hn = self.w_lstm(inputs, h0)
        output = output.squeeze(dim=0)
        h0 = hn

        logit = self.w_soft(output)
        if self.temperature > 0:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * torch.tanh(logit)

        branch_id_dist = Categorical(logits=logit)
        if determine_sample:
          branch_id = logit.argmax(dim=1)
        else:
          branch_id = branch_id_dist.sample()

        arc_seq.append(branch_id)

        log_prob = branch_id_dist.log_prob(branch_id)
        log_probs.append(log_prob.view(-1))
        entropy = branch_id_dist.entropy()
        entropys.append(entropy.view(-1))

      else:
        # https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
        assert False, "Not implemented error: search_whole_channels = False"

      # inputs = self.w_emb(branch_id)

    self.sample_arc = torch.stack(arc_seq, dim=1)

    self.sample_entropy = torch.stack(entropys, dim=1)

    self.sample_log_prob = torch.stack(log_probs, dim=1)
    self.sample_prob = self.sample_log_prob.exp()

    batched_arcs = self.sample_arc.repeat((bs, 1))
    return batched_arcs

  def train_controller(self, G, z, y, controller, controller_optim, iteration):
    """

    :param controller: for ddp training
    :return:
    """
    meter_dict = {}

    G.eval()
    controller.train()

    controller.zero_grad()

    z_samples = z.sample()
    batched_arcs = controller(bs=len(z_samples))
    sample_entropy = get_ddp_attr(controller, 'sample_entropy')
    sample_log_prob = get_ddp_attr(controller, 'sample_log_prob')

    pool_list, logits_list = [], []
    for i in range(get_ddp_attr(controller, 'num_aggregate')):
      z_samples = z.sample().to(self.device)
      y_samples = y.sample().to(self.device)
      with torch.set_grad_enabled(False):
        x = G(z=z_samples, y=y_samples, batched_arcs=batched_arcs)

      pool, logits = self.FID_IS.get_pool_and_logits(x)

      pool_list.append(pool)
      logits_list.append(logits)

    pool = np.concatenate(pool_list, 0)
    logits = np.concatenate(logits_list, 0)

    reward_g, _ = self.FID_IS.calculate_IS(logits)
    meter_dict['reward_g'] = reward_g

    # detach to make sure that gradients aren't backpropped through the reward
    reward = torch.tensor(reward_g).cuda()
    sample_entropy_mean = sample_entropy.mean()
    meter_dict['sample_entropy'] = sample_entropy_mean.item()
    reward += self.entropy_weight * sample_entropy_mean

    if self.baseline is None:
      baseline = torch.tensor(reward_g)
    else:
      baseline = self.baseline - (1 - self.bl_dec) * (self.baseline - reward)
      # detach to make sure that gradients are not backpropped through the baseline
      baseline = baseline.detach()

    sample_log_prob_mean = sample_log_prob.mean()
    meter_dict['sample_log_prob'] = sample_log_prob_mean.item()
    loss = -1 * sample_log_prob_mean * (reward - baseline)

    meter_dict['reward'] = reward.item()
    meter_dict['baseline'] = baseline.item()
    meter_dict['loss'] = loss.item()

    loss.backward(retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), self.child_grad_bound)
    meter_dict['grad_norm'] = grad_norm

    controller_optim.step()

    baseline_list = comm.all_gather(baseline)
    baseline_mean = sum(map(lambda v: v.item(), baseline_list)) / len(baseline_list)
    baseline.fill_(baseline_mean)
    self.baseline = baseline

    if iteration % self.log_every_iter == 0:
      default_dicts = collections.defaultdict(dict)
      for meter_k, meter in meter_dict.items():
        if meter_k in ['reward', 'baseline']:
          default_dicts['reward_baseline'][meter_k] = meter
        else:
          default_dicts[meter_k][meter_k] = meter
      summary_defaultdict2txtfig(default_dict=default_dicts,
                                 prefix='train_controller',
                                 step=iteration,
                                 textlogger=self.myargs.textlogger)
    comm.synchronize()
    return