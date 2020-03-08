import torch
from torch import  distributions

from template_lib.utils import get_attr_kwargs
from .build import D2DISTRIBUTIONS_REGISTRY


def fairnas_repeat_tensor(num_subpath, sample):
  repeat_arg = [1] * (sample.dim() + 1)
  repeat_arg[1] = num_subpath
  sample = sample.unsqueeze(1).repeat(repeat_arg)
  sample = sample.view(-1, *sample.shape[2:])
  return sample


@D2DISTRIBUTIONS_REGISTRY.register()
class FairNASNormal(distributions.normal.Normal):

  def __init__(self, cfg, **kwargs):

    loc                                 = get_attr_kwargs(cfg, 'loc', default=0, **kwargs)
    scale                               = get_attr_kwargs(cfg, 'scale', default=1, **kwargs)
    self.sample_shape                   = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)
    self.num_ops                        = get_attr_kwargs(cfg, 'num_ops', **kwargs)

    super(FairNASNormal, self).__init__(loc=loc, scale=scale)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(FairNASNormal, self).sample(sample_shape=self.sample_shape)
      sample = fairnas_repeat_tensor(num_subpath=self.num_ops, sample=sample)
    else:
      sample = super(FairNASNormal, self).sample(sample_shape=sample_shape)
    return sample


@D2DISTRIBUTIONS_REGISTRY.register()
class FairNASCategoricalUniform(distributions.Categorical):

  def __init__(self, cfg, **kwargs):

    n_classes                             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    validate_args                         = get_attr_kwargs(cfg, 'validate_args', default=None, **kwargs)
    self.sample_shape                     = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)
    self.num_ops                          = get_attr_kwargs(cfg, 'num_ops', **kwargs)

    if isinstance(self.sample_shape, int):
      self.sample_shape = [self.sample_shape, ]

    probs = torch.ones(n_classes) * 1./n_classes
    super(FairNASCategoricalUniform, self).__init__(probs=probs, logits=None, validate_args=validate_args)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(FairNASCategoricalUniform, self).sample(sample_shape=self.sample_shape)
      sample = fairnas_repeat_tensor(num_subpath=self.num_ops, sample=sample)
    else:
      sample = super(FairNASCategoricalUniform, self).sample(sample_shape=sample_shape)
    return sample