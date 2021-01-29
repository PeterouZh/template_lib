import yaml
from easydict import EasyDict
import logging
import sys
import functools
import os

import torch


from detectron2.utils import comm
from template_lib.v2.GAN.evaluation.pytorch_FID_IS_score import PyTorchFIDISScore
from template_lib.v2.ddp import ddp_init
from template_lib.v2.config_cfgnode import global_cfg





def run():
  from template_lib.d2.data import build_dataset_mapper
  from template_lib.d2template.trainer.base_trainer import build_detection_test_loader
  from template_lib.v2.GAN.evaluation import build_GAN_metric
  from template_lib.d2.utils.d2_utils import D2Utils
  from template_lib.d2.data import build_cifar10

  from detectron2.utils import logger
  logger.setup_logger()

  config = PyTorchFIDISScore.update_cfg(global_cfg)

  cfg = D2Utils.create_cfg()
  cfg = D2Utils.cfg_merge_from_easydict(cfg, config)

  # fmt: off
  dataset_name                 = cfg.dataset_name
  IMS_PER_BATCH                = cfg.IMS_PER_BATCH
  img_size                     = cfg.img_size
  NUM_WORKERS                  = cfg.NUM_WORKERS
  dataset_mapper_cfg           = cfg.dataset_mapper_cfg
  GAN_metric                   = cfg.GAN_metric
  # fmt: on

  cfg.defrost()
  cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
  cfg.freeze()

  num_workers = comm.get_world_size()
  batch_size = IMS_PER_BATCH // num_workers



  dataset_mapper = build_dataset_mapper(dataset_mapper_cfg, img_size=img_size)
  data_loader = build_detection_test_loader(
    cfg, dataset_name=dataset_name, batch_size=batch_size, mapper=dataset_mapper)

  FID_IS_torch = build_GAN_metric(GAN_metric)
  FID_IS_torch.calculate_fid_stat_of_dataloader(data_loader=data_loader)

  comm.synchronize()

  pass


if __name__ == '__main__':
  args = ddp_init()
  run()









