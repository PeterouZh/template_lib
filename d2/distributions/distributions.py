import torch
from torch import  distributions

from template_lib.utils import get_attr_kwargs
from .build import D2DISTRIBUTIONS_REGISTRY


@D2DISTRIBUTIONS_REGISTRY.register()
class Normal(distributions.normal.Normal):

  def __init__(self, cfg, **kwargs):

    loc                                 = get_attr_kwargs(cfg, 'loc', default=0, **kwargs)
    scale                               = get_attr_kwargs(cfg, 'scale', default=1, **kwargs)
    self.sample_shape                   = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)

    super(Normal, self).__init__(loc=loc, scale=scale)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(Normal, self).sample(sample_shape=self.sample_shape)
    else:
      sample = super(Normal, self).sample(sample_shape=sample_shape)
    return sample


@D2DISTRIBUTIONS_REGISTRY.register()
class Categorical(distributions.Categorical):

  def __init__(self, cfg, **kwargs):

    probs                                 = get_attr_kwargs(cfg, 'probs', default=None, **kwargs)
    logits                                = get_attr_kwargs(cfg, 'logits', default=None, **kwargs)
    validate_args                         = get_attr_kwargs(cfg, 'validate_args', default=None, **kwargs)
    self.sample_shape                     = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)

    if isinstance(self.sample_shape, int):
      self.sample_shape = [self.sample_shape, ]

    super(Categorical, self).__init__(probs=probs, logits=logits, validate_args=validate_args)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(Categorical, self).sample(sample_shape=self.sample_shape)
    else:
      sample = super(Categorical, self).sample(sample_shape=sample_shape)
    return sample


@D2DISTRIBUTIONS_REGISTRY.register()
class CategoricalUniform(distributions.Categorical):

  def __init__(self, cfg, **kwargs):

    n_classes                             = get_attr_kwargs(cfg, 'n_classes', **kwargs)
    validate_args                         = get_attr_kwargs(cfg, 'validate_args', default=None, **kwargs)
    self.sample_shape                     = get_attr_kwargs(cfg, 'sample_shape', default=None, **kwargs)

    if isinstance(self.sample_shape, int):
      self.sample_shape = [self.sample_shape, ]

    probs = torch.ones(n_classes) * 1./n_classes
    super(CategoricalUniform, self).__init__(probs=probs, logits=None, validate_args=validate_args)
    pass

  def sample(self, sample_shape=None):
    if sample_shape is None:
      sample = super(CategoricalUniform, self).sample(sample_shape=self.sample_shape)
    else:
      sample = super(CategoricalUniform, self).sample(sample_shape=sample_shape)
    return sample