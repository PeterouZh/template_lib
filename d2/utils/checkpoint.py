import sys
import torch
from torch import nn

from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import Checkpointer, PeriodicCheckpointer


class DumpModule(nn.Module):
  def __init__(self, model_dict):
    super(DumpModule, self).__init__()
    for name, model in model_dict.items():
      setattr(self, name, model)
    pass


class D2Checkpointer(object):

  def __init__(self, model_dict, optim_dict, ckptdir,
               period, max_to_keep=5, maxsize=sys.maxsize, state_dict=None):

    self.period = period
    self.max_to_keep = max_to_keep
    self.maxsize = maxsize

    self.state_dict = state_dict if state_dict is not None else {}

    self.logger = setup_logger(output=ckptdir, name='fvcore')

    self.checkpointer = self.get_d2_checkpointer(model_dict=model_dict, optim_dict=optim_dict, ckptdir=ckptdir)
    self.periodic_checkpointer = self.get_d2_periodic_checkpointer()
    pass

  @staticmethod
  def get_d2_checkpointer(model_dict, optim_dict, ckptdir):
    ckpt_model = DumpModule(model_dict)
    checkpointer = Checkpointer(ckpt_model, ckptdir, **optim_dict)
    return checkpointer

  def get_d2_periodic_checkpointer(self, ):
    """
    periodic_checkpointer.step(epoch, **{'first_epoch': epoch})
    periodic_checkpointer.save(name='best', **{'max_mIoU': max_mIoU})
    """
    periodic_checkpointer = PeriodicCheckpointer(
      self.checkpointer, period=self.period, max_iter=self.maxsize, max_to_keep=self.max_to_keep)
    return periodic_checkpointer

  def step(self, itr, **kwargs):
    self.periodic_checkpointer.step(itr, **self.state_dict, **kwargs)

  def save(self, name, **kwargs):
    self.periodic_checkpointer.save(name=name, **self.state_dict, **kwargs)