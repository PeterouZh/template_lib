import copy
import functools
import logging
import os

import tqdm
import collections
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import numpy as np

from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage

from template_lib.d2.distributions.fairnas_noise_sample import fairnas_repeat_tensor
from template_lib.d2.distributions import build_d2distributions
from template_lib.d2.models import build_d2model
from template_lib.trainer.base_trainer import Trainer
from template_lib.trainer import get_ddp_attr
from template_lib.gans import inception_utils, gan_utils, gan_losses, GANLosses
from template_lib.gans.networks import build_discriminator, build_generator
from template_lib.gans.evaluation import get_sample_imgs_list_ddp
from template_lib.d2.optimizer import build_optimizer
from template_lib.utils import modelarts_utils
from template_lib.gans.evaluation import build_GAN_metric_dict
from template_lib.gans.evaluation.fid_score import FIDScore
from template_lib.gans.models import build_GAN_model
from template_lib.utils import get_eval_attr, print_number_params, get_attr_kwargs, get_attr_eval

from .build import TRAINER_REGISTRY


class DumpModule(nn.Module):
  def __init__(self, model_dict):
    super(DumpModule, self).__init__()
    for name, model in model_dict.items():
      setattr(self, name, model)
    pass


@TRAINER_REGISTRY.register()
class BaseTrainer(nn.Module):

    def __init__(self, cfg, myargs, iter_every_epoch, **kwargs):
      super().__init__()

      self.cfg                           = cfg
      self.myargs                        = myargs
      self.iter_every_epoch              = iter_every_epoch

      self.device = torch.device(f'cuda:{comm.get_rank()}')
      self.logger = logging.getLogger('tl')
      self.distributed = comm.get_world_size() > 1

      torch.cuda.set_device(self.device)
      self.build_models(cfg=cfg)
      self.to(self.device)

    def build_models(self, cfg):
      self.models = {}
      self.optims = {}

      self._print_number_params(self.models)

    def build_optimizer(self):

      optims_dict = self.optims

      return optims_dict

    def get_saved_model(self):
      models = {}
      for name, model in self.models.items():
        if isinstance(model, DistributedDataParallel):
          models[name] = model.module
        else:
          models[name] = model
      saved_model = DumpModule(models)
      return saved_model

    def train_func(self, data, iteration, pbar):
      """Perform architecture search by training a controller and shared_cnn.
      """
      if comm.is_main_process() and iteration % self.iter_every_epoch == 0:
        pbar.set_postfix_str(s="BaseTrainer ")

      images, labels = self._preprocess_image(data)
      images = images.tensor

      comm.synchronize()

    def _get_bs_per_worker(self, bs):
      num_workers = get_world_size()
      bs_per_worker = bs // num_workers
      return bs_per_worker

    def _get_tensor_of_main_processing(self, tensor):
      tensor_list = comm.all_gather(tensor)
      tensor = tensor_list[0].to(self.device)
      return tensor

    def _preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        labels = torch.LongTensor([x["label"] for x in batched_inputs]).to(self.device)
        images = ImageList.from_tensors(images)
        return images, labels

    def _get_ckpt_path(self, ckpt_dir, ckpt_epoch, iter_every_epoch):
      eval_iter = (ckpt_epoch) * iter_every_epoch - 1
      eval_ckpt = os.path.join(ckpt_dir, f'model_{eval_iter:07}.pth')
      self.logger.info(f'Load weights:\n{os.path.abspath(eval_ckpt)}')
      return eval_ckpt

    def _print_number_params(self, models_dict):
      print_number_params(models_dict=models_dict)

